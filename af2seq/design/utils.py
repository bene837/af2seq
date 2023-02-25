import json
import subprocess
import logging
import time
import os
from tempfile import TemporaryDirectory

import jax
import tmscoring
from simtk import unit
from simtk.openmm import app as openmm_app
from simtk.openmm.app.internal.pdbstructure import PdbStructure
import io
# from pymol import cmd
from af2seq.alphafold.relax import cleanup
from af2seq.alphafold.relax import amber_minimize
from af2seq.alphafold.common import protein
from af2seq.alphafold.common import residue_constants
import numpy as np
from Bio import PDB
import pydssp
import torch


ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms


def get_logger(loggername, level="INFO", file=None):
    """
    generates and returns a logger for a script
    :param loggername: name of the logger
    :param level: verbosity of the logger
    :param file: file to save logs to
    :return: logger
    """

    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    logger = logging.getLogger(loggername)

    logger.setLevel(level)

    if file is not None:
        os.makedirs(f"mainlogs/{file}", exist_ok=True)
        path = f"mainlogs/{file}/{time.time()}.log"
        file_handler = logging.FileHandler(path)
        formatter = logging.Formatter(log_format, datefmt="%m/%d/%Y %I:%M:%S %p")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"logging to {path}")
    # Don't forget to add the file handler

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format, datefmt="%m/%d/%Y %I:%M:%S %p")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def encode(seq: str):
    pssm = np.zeros((len(seq), 20))
    revmap = {
        val: nr
        for nr, val in enumerate(
            [
                "A",
                "R",
                "N",
                "D",
                "C",
                "Q",
                "E",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
                "X",
            ]
        )
    }

    for nr, i in enumerate(seq):
        pssm[nr, revmap[i]] = 1  # 0.01

    return pssm


def decode(seq):
    revmap = {
        nr: val
        for nr, val in enumerate(
            [
                "A",
                "R",
                "N",
                "D",
                "C",
                "Q",
                "E",
                "G",
                "H",
                "I",
                "L",
                "K",
                "M",
                "F",
                "P",
                "S",
                "T",
                "W",
                "Y",
                "V",
                "X",
            ]
        )
    }

    return "".join([revmap[i] for i in np.asarray(seq)])


def tm_align(pdb_1, pdb_2):
    """Takes two pdb structures and calculates the TM alignment score"""

    output = subprocess.check_output(["/work/lpdi/bin/TMalign", pdb_1, pdb_2])
    lines = output.splitlines()

    for line in lines:
        line_str = str(line)
        if line_str[2:-1].startswith("Aligned length"):
            a_length = line_str[2:-1].split()[2][:-1]
            RMSD = line_str[2:-1].split()[4][:-1]
        elif line_str[2:-1].startswith("TM-score="):
            TM = line_str[2:-1].split()[1]
        elif line_str[2:-1].startswith("Length of Chain_1"):
            L = line_str[2:-1].split()[3]

    return float(TM), float(RMSD), float(a_length), int(L)


def overwrite_b_factors(pdb_str: str, bfactors: np.ndarray) -> str:
    """From the original AF2 github: Overwrites the B-factors in pdb_str with contents of bfactors array."""

    parser = PDB.PDBParser(QUIET=True)
    handle = io.StringIO(pdb_str)
    structure = parser.get_structure('', handle)

    curr_resid = ('', '', '')
    idx = -1
    for atom in structure.get_atoms():
        atom_resid = atom.parent.get_id()
        if atom_resid != curr_resid:
            idx += 1
            if idx >= bfactors.shape[0]:
                raise ValueError('Index into bfactors exceeds number of residues. '
                                 'B-factors shape: {shape}, idx: {idx}.')
        curr_resid = atom_resid
        atom.bfactor = bfactors[idx, residue_constants.atom_order['CA']]

    new_pdb = io.StringIO()
    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(new_pdb)
    return new_pdb.getvalue()


def cleanup_for_json(dictionary):
    output = {}
    for key, value in dictionary.items():

        if isinstance(value, dict):
            value = cleanup_for_json(value)
        elif isinstance(value, np.ndarray or isinstance(value, jnp.DeviceArray)):
            value = value.tolist()
        try:
            json.dumps(value)
            output[key] = value
        except TypeError:

            continue

    return output


def amber_relax(pred_prot):
    pdb_str = protein.to_pdb(pred_prot)
    pdb_file = io.StringIO(pdb_str)
    alterations_info = {}
    fixed_pdb = cleanup.fix_pdb(pdb_file, alterations_info)
    fixed_pdb_file = io.StringIO(fixed_pdb)
    pdb_structure = PdbStructure(fixed_pdb_file)
    cleanup.clean_structure(pdb_structure, alterations_info)

    as_file = openmm_app.PDBFile(pdb_structure)
    pdb_string = amber_minimize._get_pdb_string(
        as_file.getTopology(), as_file.getPositions()
    )

    exclude_residues = []
    tolerance = 2.39
    stiffness = 10.0
    # Assign physical dimensions.
    tolerance = tolerance * ENERGY
    stiffness = stiffness * ENERGY / (LENGTH ** 2)

    max_iterations = 0

    ret = amber_minimize._openmm_minimize(
        pdb_string,
        max_iterations=max_iterations,
        tolerance=tolerance,
        stiffness=stiffness,
        restraint_set="non_hydrogen",
        exclude_residues=exclude_residues,
    )

    min_pdb = ret["min_pdb"]
    min_pdb = overwrite_b_factors(min_pdb, pred_prot.b_factors)

    return min_pdb


def score_alignment(feat, result0, target_file):
    """Takes in the result of an alphafold prediction and makes a TM-alignment. Then returns the corresponing TM-score and RMSD."""

    # write out the latest pdb file so a TM-alignment can be generated.
    with TemporaryDirectory(prefix="temp") as tmpdir:
        f = open(tmpdir + "/temp_struct.pdb", "w")
        f.write(protein.to_pdb(protein.from_prediction(feat, result0)))
        f.close()

        # Align the structures
        alignment = tmscoring.TMscoring(
            target_file, tmpdir + "/temp_struct.pdb"
        )
        alignment.optimise()

    # Get the TM score
    TM_aligned = alignment.tmscore(**alignment.get_current_values())

    # RMSD of the protein aligned according to TM-based alignment
    RMSD_aligned = alignment.rmsd(**alignment.get_current_values())

    return TM_aligned, RMSD_aligned


def extract_metrics(result, feat, target_file):
    # obtain the results of the 5 different models
    TM_aligned_mean = []
    RMSD_aligned_mean = []

    for i in range(len(result)):
        result0 = jax.tree_map(lambda x: x[i], result)
        TM_aligned, RMSD_aligned = score_alignment(feat, result0, target_file)
        TM_aligned_mean.append(TM_aligned)
        RMSD_aligned_mean.append(RMSD_aligned)

    TM_aligned_mean = np.mean(TM_aligned_mean)
    RMSD_aligned_mean = np.mean(RMSD_aligned_mean)
    return TM_aligned_mean, RMSD_aligned_mean


def calc_seqsim(seq_1, seq_2):
    c = 0
    for i in range(len(seq_1)):
        if seq_1[i] == seq_2[i]:
            c += 1

    seqsim = (c / len(seq_1)) * 100
    return seqsim


def generate_start_sequence(file):
    """Generates a starting sequence using secondary structure annotation from dssp"""
    with open(file, 'r') as f:
        pdb = f.read()

    coord = torch.tensor(pydssp.read_pdbtext(pdb))
    dssp = pydssp.assign(coord, out_type='c3')
    sec_struct = ''.join(dssp)

    org = "HE-"
    trans = "AVG"

    translation = str.maketrans(org, trans)

    return sec_struct.translate(translation)
