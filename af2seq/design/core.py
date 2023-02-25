import json
import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import List
import optax
import time
import numpy as np
import jax
from jax import value_and_grad
import jax.numpy as jnp

try:
    from pyrosetta import *
except ImportError:
    print('Warning: Couldnt find pyrosetta. Some features will not work properly')
from af2seq.alphafold.notebooks import notebook_utils

# Load correct tqdm progress bar for terminal or jupyter notebook
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

from af2seq.alphafold.common import protein
from af2seq.alphafold.data import pipeline_multimer
from ..alphafold.data import parsers, feature_processing, msa_pairing, pipeline, templates
from ..alphafold.model import geometry, all_atom_multimer, design_model as design, config, data, modules
from ..alphafold.common import confidence
from .pre_processing import build_msa, build_msa_ptm, build_pssm
from .utils import decode, get_logger, cleanup_for_json, extract_metrics, calc_seqsim
from .loss_functions import LOSSES

import warnings

warnings.filterwarnings("ignore")


class Design:
    """Class contains functions to run AlphaFold optimization trajectories."""

    def __init__(self, datadir: str, output_path: str = None, debug: bool = False):
        """

        Args:
            datadir: (str) path to the alphafold weights
            output_path: (str) path were results will be saved
            debug: (bool) enables additional logging output
        """


        self.DATADIR = datadir

        self.models = None
        self.modeltype = None
        self.max_iter = None
        self.startseq = None
        self.gt_seq = None
        self.clamped = None

        self.ident = f'{int(time.time())}{os.getpid()}'
        self.log = get_logger(str(self.ident), level="INFO" if not debug else "DEBUG")
        self.log.propagate = False
        self.name = None
        # CONFIG
        self.counter = 0
        self.best_result = {"sequence": "No run started!"}
        self.best_loss = np.inf
        self.pssm = None
        self.chainmask = None
        self.lr = None
        self.loss_track = {
            "learning_rate": [],
            "tm": [],
            "rmsd": [],
            "norm": [],
            "sequences": [],
            "mean": [],
            "upper": [],
            "lower": [],
            "plddt": [],
            "ptm": [],
            "seq_sim": []
        }
        self.start_time = None
        self.msas = None
        self.chain_length = None
        self.output_path = output_path
        self.target_file = None
        self.track_file_created = False
        self.tracking_file = None
        self.mode = 'gd'
        self.pdbs = []
        self.model_loaded = False
        self.encode_start_value = 0.01
        self.loss_fn = {}
        self.temp_paths = None
        self.use_msas = False

    def __str__(self):
        """Print info about the best scored sequence and the number of iterations in the
        optimization trajectory."""
        return f'[INFO]:\n\t[Best sequence]: {self.best_result["sequence"]}\n\t[Iterations]: {self.counter}'

    def _prep_model(self):
        """Prepares the AlphaFold model."""

        def alphafold_fn(target, processed_feature_dict, model_params):
            """Wrap AlphaFold and specify the loss function for backpropagation."""
            feat = {**processed_feature_dict}
            feat.update(target)
            result, _ = self.model.apply(model_params, jax.random.PRNGKey(0), feat)
            result["design_loss"] = {}
            overall_loss = 0.0
            for k, l_fn in self.loss_fn.items():
                loss = l_fn(result)
                result["design_loss"][k] = loss
                overall_loss = overall_loss + jnp.mean(loss)

            return overall_loss, result

        # Compile forward and backward pass
        alphafold_fwd_bwd = jax.jit(
            jax.vmap(
                value_and_grad(alphafold_fn, has_aux=True),
                in_axes=(None, None, 0),
            )
        )
        return alphafold_fwd_bwd

    def _prep_opt(self):
        """Initializes the ADAM optimizer."""
        self.optimizer = optax.adam(self.lr)
        self.opt_state = self.optimizer.init({"tf": self.pssm[self.chainmask]})

    def _select_and_load_params(self, modeltype: str, recycles: int):
        """
        Loads the AlphaFold model parameters and  modify the configuration.
        Args:
            modeltype: (str) ptm or multimer
            recycles: (int) number of recycles used in alphafold

        Returns: model configuration, parameter for one model and parameter for all others

        """
        if modeltype == "multimer":
            self.log.info("Using model 3,4,5")

            model_config = config.model_config("model_5_multimer")
            model_config.num_ensemble_eval = 1
            model_config.model.global_config.multimer_mode = True
            model_config.resample_msa_in_recycling = True  # was false
            model_config.model.resample_msa_in_recycling = True  # was false
            model_config.model.global_config.use_remat = True
            model_config.model.global_config.deterministic = False  # WAS TRUE
            model_config.model.global_config.subbatch_size = (
                None  # BETA, should speed things up for backprop
            )
            if self.use_msas:
                model_config.model.embeddings_and_evoformer.num_msa = 512  # was 512
            else:
                model_config.model.embeddings_and_evoformer.num_msa = 1

            model_config.model.embeddings_and_evoformer.num_extra_msa = 1
            model_config.model.num_recycle = recycles

            # LOAD PARAMS
            single_params = data.get_model_haiku_params(
                model_name=f"model_5_multimer", data_dir=self.DATADIR
            )
            all_params = [single_params]
            for i in range(3, 5):
                model_name = f"model_{i}_multimer"
                params = data.get_model_haiku_params(
                    model_name=model_name, data_dir=self.DATADIR
                )
                all_params.append({k: params[k] for k in single_params.keys()})

        elif modeltype == "ptm":
            if self.temp_paths is not None:
                self.log.info("Using model 1 and 2 (using templates)")
                model_config = config.model_config("model_1_ptm")  # model 1 and 2 use templates
                single_params = data.get_model_haiku_params(model_name=f"model_1_ptm", data_dir=self.DATADIR)
                all_params = [single_params]
                for i in range(2, 3):
                    model_name = f"model_{i}_ptm"
                    params = data.get_model_haiku_params(model_name=model_name, data_dir=self.DATADIR)
                    all_params.append({k: params[k] for k in single_params.keys()})

            else:
                self.log.info(f"Using models: {self.models}")
                model_config = config.model_config("model_5_ptm")
                single_params = data.get_model_haiku_params(model_name=f"model_5_ptm", data_dir=self.DATADIR)
                all_params = []
                for i in self.models:
                    model_name = f"model_{i}_ptm"
                    params = data.get_model_haiku_params(model_name=model_name, data_dir=self.DATADIR)
                    all_params.append({k: params[k] for k in single_params.keys()})

            model_config.model.global_config.multimer_mode = False
            model_config.data.eval.num_ensemble = 1
            model_config.data.common.resample_msa_in_recycling = False
            model_config.data.common.reduce_msa_clusters_by_max_templates = False
            model_config.model.resample_msa_in_recycling = False
            model_config.model.global_config.use_remat = True
            model_config.model.global_config.subbatch_size = (
                None  # BETA, should speed things up
            )
            model_config.model.global_config.deterministic = False  # set to False for design
            if self.use_msas:
                # These values are used in the intial training of AF
                # (see supplementary information Jumper et al. 2021.)
                model_config.data.eval.max_msa_clusters = 32  # was 512
                model_config.data.common.max_extra_msa = (
                    1  # was 5120 (and 1024 in training)
                )

            else:
                model_config.data.eval.max_msa_clusters = 1
                model_config.data.common.max_extra_msa = 1

            model_config.data.common.num_recycle = recycles
            model_config.model.num_recycle = recycles

        else:
            raise ValueError(f"No valid mode: {modeltype}")

        model_params = jax.tree_map(lambda *m: jnp.stack(m, axis=0), *all_params)

        return model_config, single_params, model_params

    def _configure_model(self, modeltype, recycles):
        """Configure the model and initiating it"""
        (
            self.model_config,
            single_params,
            self.model_params,
        ) = self._select_and_load_params(modeltype, recycles)
        self.model = design.RunModelDesign(self.model_config, single_params)

    def set_loss(self, loss):
        """Takes the mode and loss function and sets the correct starting loss for optimization."""
        for l, weight in loss.items():
            if l.lower() not in (
                    "fape",
                    "ptm",
                    "plddt",
                    "tm",
                    'hotspot',
                    'distogram'
            ):
                raise ValueError(
                    "Not a valid loss! Please choose one of the following: "
                    "('fape','ptm','plddt','tm','hotspot',distogram)"
                )
            if self.mode == "gd" and l.lower() == "tm":
                raise NotImplemented(
                    "This loss is currently not supported for gradient descent"
                )
            if l == "tm" and len(self.chains) > 1:
                raise ValueError("WARNING: TM-score only works for monomer design!!!")
            self.loss_fn[l.lower() + "_loss"] = partial(LOSSES[l.lower() + "_loss"], weight=weight)

    def _get_msas(self, start_seq, msas):
        if msas is not None and self.modeltype == "ptm":
            msa = build_msa_ptm(msas, self.chain_length, start_seq)
            return [msa]
        else:
            return build_msa(self.log, start_seq, msas)

    def _build_templates(self, sequence):
        if self.startseq is None:
            raise ValueError("Please specify a start sequence to enable templates")

        startlen = len(self.startseq)
        if startlen == 1:
            seq_1 = sequence
            msa = parsers.Msa([sequence], [[0] * len(sequence)], ["query"])
        elif startlen == 2:
            self.log.info(f"EXPERIMENTAL: Found templates. Using them instead of MSA")

            seq_1 = self.startseq[0] + '-' * len(self.startseq[1])
            seq_2 = '-' * len(self.startseq[0]) + self.startseq[1]
            msa = parsers.Msa([seq_1, seq_2], [[0] * len(seq_1), [0] * len(seq_2)], ["query1", "query2"])
        else:
            raise NotImplementedError('Templates only work for 2 sequences')

        # Get inputs
        name = self.temp_paths.split('/')[-1]
        template_path = '/'.join(self.temp_paths.split('/')[:-1])

        # Create template features
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=template_path,
            max_template_date="2100-01-01",
            max_hits=20,
            kalign_binary_path="kalign",
            release_dates_path=None,
            obsolete_pdbs_path=None)

        index = 1
        aligned_cols = len(sequence)  # TRY: len(unjoined_sequences[0]) if it doesnt work
        sum_probs = 100.0
        query = seq_1  # sequences[0]
        hit_sequence = seq_1  # sequences[0]
        indices_query = list(range(aligned_cols))
        indices_hit = list(range(aligned_cols))

        hit = parsers.TemplateHit(index, name, aligned_cols, sum_probs, query,
                                  hit_sequence, indices_query, indices_hit)

        temps = template_featurizer.get_templates(seq_1, [hit])
        return msa, temps.features

    def design(
            self,
            target_file: str,
            start_seq: list = None,
            chains: list = None,
            msas: list = None,
            filename: str = None,
            iterations: int = 500,
            recycles: int = 0,
            lr: float = 1e-3,
            clampval: float = 0.0,
            aa_mask: list = tuple("C"),
            fix_pos: list = None,
            disable_loss_pos: list = None,
            enable_sc_loss: list = None,
            modeltype: str = None,
            loss: dict = None,
            models: list = None,
            template_path: str = None
    ):
        """
        Runs the gradient descent pipeline.
        Args:
            target_file: string to the pdb file to use as a target
            start_seq: list of initial sequence(s) to start optimization trajectory.
            chains: list of which chains to design, zero indexed e.g. [1] designs the second chain of the start_seq.
            msas: list of paths to avaible MSAs. If no MSA available input None e.g. [None, msa_path_1]
            filename: string of the name of the run.
            iterations: number of iterations of gradient descent/mcmc optimization.
            recycles: number of recycles to use.
            lr: initial learning rate of ADAM optimizer.
            clampval: clamp value for FAPE loss.
            aa_mask: which amino acids to mask in the PSSM. These amino acids will not be used for design.
            fix_pos: list of residue positions to fix in the initial sequence (start_seq).
            disable_loss_pos: list of residue positions not to include in the backbone FAPE loss.
            enable_sc_loss: list of residue positions for which to include the sidechain FAPE loss.
            modeltype: string for which AlphaFold model to use. AlphaFold ptm 'ptm' or AlphaFold Multimer 'multimer'.
            loss: (dict) losses that are used for optimization. str list or dict {loss:weight}
            models: (list) alphafold models that are run simultaneously.
            template_path (str) EXPERIMENTAL

        Returns:

        """

        if models is None:
            self.models = [5, 4, 3, 2, 1]
        else:
            self.models = models

        if loss is None:
            loss = {"fape": 1}
        elif isinstance(loss, list) or isinstance(loss, tuple):
            loss = {l: 1 for l in loss}
        elif isinstance(loss, str):
            loss = {loss: 1}

        # SET THE CORRECT LOSSES
        self.set_loss(loss)
        if self.start_time is None:
            self.start_time = time.asctime()

        self.target_file = target_file
        self.lr = lr
        self.clamped = clampval
        self.startseq = start_seq
        self.name = filename
        self.max_iter = iterations
        self.temp_paths = template_path

        # CHECK INPUT AND BUILD PSSM:
        with open(target_file, "r") as f:
            pdb = f.readlines()

        target_prot = protein.from_pdb_string("".join(pdb))

        # CHECK MODELTYPE
        if modeltype is None:
            gtchains, _ = np.unique(target_prot.chain_index, return_counts=True)
            self.log.debug(f"chains {len(gtchains)}")
            modeltype = "ptm" if len(gtchains) == 1 else "multimer"
        self.modeltype = modeltype
        self.log.info(f"Using {self.modeltype} model")

        (start_seq, self.gt_seq, self.pssm, self.chainmask, self.positional_mask, self.sidechain_mask,
         self.chains, self.chain_length, self.mcmc_mask) = build_pssm(self.log,
                                                                      target_prot,
                                                                      start_seq,
                                                                      chains,
                                                                      aa_mask,
                                                                      fix_pos,
                                                                      disable_loss_pos,
                                                                      enable_sc_loss,
                                                                      encode_value=self.encode_start_value,
                                                                      mode=self.mode)
        # If multichain start sequence merge it to one single input.
        if modeltype == "ptm" and isinstance(start_seq, list) and len(start_seq) > 1:
            self.log.info("Running ptm model with more than 1 chain. Merging sequences!")
            start_seq = ["".join(start_seq)]
        # self.startseq = start_seq

        self.log.debug(f"Startseq: {self.startseq}, Modified: {start_seq}")
        # check if msa list has been provided.
        self.msas = self._get_msas(start_seq, msas)

        self._configure_model(self.modeltype, recycles)

        alphafold_model = self._prep_model()

        # GENERATE GROUNDTRUTH (GT) FEAT:
        gt_dict = self._preprocess_gt(target_prot)
        feat = self.feat_update(start_seq)
        feat.update(gt_dict)

        # INTRODUCE CHAINBREAK IN PTM MULTIMER PREDICTION
        if modeltype == "ptm" and len(self.chains) > 1:
            feat["residue_index"] = self._chainbreak(feat)

        self.model.init_params(feat)
        self._prep_opt()

        self.log.info(f"Starting design:")
        self.log.info(f"[Target]: {target_file}")

        self._create_logging_files()

        # MAIN LOOP
        pbar = tqdm(range(iterations))
        self.log.debug(feat.keys())
        for nr in pbar:
            sequence, loss, highest_confidence_result, feat, plddt, ptm = self.step(alphafold_model, gt_dict)
            self._update_and_track(sequence, loss, highest_confidence_result, feat, plddt, ptm)
            self._track_step(nr)
            self.counter += 1
            pbar.set_postfix({"cl": loss, "bl": self.best_loss})

    def _pssm_to_sequence(self) -> List[str]:
        """
        Gets maximum likely amioacids and generates the  sequence according to the pssm

        Returns: (List[str]) sequences

        """
        prev = 0
        sequences = []
        for l in self.chain_length:
            sequences.append(
                decode(jnp.argmax(jax.nn.softmax(self.pssm[prev: prev + l]), 1))
            )
            prev += l
        return sequences

    def _chainbreak(self, feat):
        """Shift the residue index for ptm models when multiple sequences are used"""
        # Determine the chain breaks
        chain_break = []
        for i in range(len(self.chain_length) - 1):
            if i == 0:
                chain_break.append(self.chain_length[i])
            else:
                chain_break.append(sum(self.chain_length[: i + 1]))

        residue_index = np.copy(feat["residue_index"])
        for j in chain_break:
            for i in feat["residue_index"][0]:
                if j <= i:
                    residue_index[0][i] += 200

        return residue_index

    def step(self, model, gt_dict):
        """Main function describing one model step"""

        raise NotImplemented

    def _update_and_track(self, sequence, loss, highest_confidence_result, feat, plddt, ptm):
        """
        Track different metrics across the design process and save best results
        Args:
            loss:
            highest_confidence_result:
            feat:
            plddt:
            ptm:

        Returns:

        """

        if self.modeltype == "ptm":
            self.pdbs.append(protein.to_pdb(
                protein.from_prediction(feat, highest_confidence_result, np.stack([plddt for i in range(37)], axis=1))))
        else:

            self.pdbs.append(
                protein.to_pdb(
                    protein.from_prediction(
                        feat, highest_confidence_result, np.stack([plddt for i in range(37)], axis=1),
                        remove_leading_feature_dimension=False
                    )
                )
            )

        if loss < self.best_loss:
            self.best_result = {
                "pssm": self.pssm.copy(),
                "sequence": sequence,
                "plddt": plddt,
                "ptm": ptm,
                "feat": feat,
                "result": highest_confidence_result,
            }
            self.best_loss = loss

    def _calc_scores(self, result, feat, loss):
        # Calculate TM-score and RMSD
        if len(self.chains) == 1:
            TM_aligned_mean, RMSD_aligned_mean = extract_metrics(result, feat, self.target_file)
            self.loss_track["tm"].append(TM_aligned_mean)
            self.loss_track["rmsd"].append(RMSD_aligned_mean)

        # track losses
        self.loss_track["mean"].append(float(jnp.mean(loss)))
        self.loss_track["upper"].append(float(jnp.max(loss)))
        self.loss_track["lower"].append(float(jnp.min(loss)))

        loss = jnp.mean(loss)

        design_losses = jax.tree_map(lambda x: jnp.mean(x), result['design_loss'])
        for key, value in design_losses.items():
            if self.loss_track.get(key) is None:
                self.loss_track[key] = [float(value)]
            else:
                self.loss_track[key].append(float(value))

        plddt = confidence.compute_plddt(result["predicted_lddt"]["logits"])
        self.loss_track["plddt"].append(plddt.tolist())
        mean_plddt = jnp.mean(plddt, -1)

        highest_confidence = mean_plddt.argmax()
        highest_confidence_result = jax.tree_map(lambda x: x[highest_confidence], result)

        ptms = []
        for logits, breaks in zip(result["predicted_aligned_error"]["logits"],
                                  result["predicted_aligned_error"]["breaks"]):
            ptms.append(confidence.predicted_tm_score(
                logits=logits,
                breaks=breaks,
                asym_id=None))

        self.loss_track["ptm"].append(ptms)

        ptm = float(ptms[highest_confidence])

        highest_confidence_plddt = plddt[highest_confidence]

        return highest_confidence_result, ptm, highest_confidence_plddt, loss

    def _update_pssm(self, grads):
        """
        Update the PSSM with a gradient. gradient is first normalised.
        Args:
            grads:

        Returns:

        """

        self.log.debug(grads["target_feat"].shape)

        if self.modeltype == "ptm":
            mean_grad = jnp.mean(grads["target_feat"][:, 0, :, 1:21], 0)
        else:
            mean_grad = jnp.mean(grads["target_feat"][:, :, :20], 0)

        tf = (mean_grad
              / jnp.linalg.norm(mean_grad, axis=1)[0])  # DIMS: (Models,nr_seq,AAs,AA_type)

        grad_dict = {"tf": tf[self.chainmask]}

        prev = jnp.argmax(jax.nn.softmax(self.pssm), 1).copy()

        while True:

            updates, self.opt_state = self.optimizer.update(grad_dict, self.opt_state)
            self.pssm[self.chainmask] = optax.apply_updates(
                {"tf": self.pssm[self.chainmask]}, updates)["tf"]
            self.log.debug(
                f"PSSM: {prev}, NEW: {jnp.argmax(jax.nn.softmax(self.pssm), 1)}"
            )
            if jnp.any(jnp.argmax(jax.nn.softmax(self.pssm), 1) != prev):
                break

    def _create_logging_files(self):
        if self.output_path is not None:
            fname = (
                self.name
                if self.name is not None
                else f"design_{self.target_file.split('/')[-1].split('.pdb')[0]}"
            )
            out_file = f"{fname}_{self.ident}_config.json"
            file = os.path.join(self.output_path, out_file)
            out_file2 = f"{fname}_{self.ident}_tracking.csv"
            self.tracking_file = os.path.join(self.output_path, out_file2)

            cur_time = time.asctime()

            outdict = {"run_start_time": cur_time}
            outdict.update(cleanup_for_json(self.__dict__))
            with open(file, "w") as f:
                json.dump(outdict, f)

        else:
            self.log.info('No output directory. Logging to file disabled!')
            # print('No output directory. Logging to file disabled!')

    def _track_step(self, nr):
        if self.tracking_file is None:
            return
        cur_time = time.asctime()
        if not self.track_file_created:
            with open(self.tracking_file, "w") as f:
                header = []
                for h, value in self.loss_track.items():

                    if len(value) > 0:
                        if isinstance(value[0], list):

                            for i in range(len(value[0])):
                                header.append(f"{h}_{i + 1}")

                        else:
                            header.append(h)
                    else:
                        header.append(h)

                header.append("Time")
                f.write('Step,')
                f.write(','.join(header))
                f.write('\n')
            self.track_file_created = True

        with open(self.tracking_file, "a") as f:

            values = []
            for key, val in self.loss_track.items():

                if key == 'sequences':
                    values.append(':'.join(val[-1]))
                elif len(val) == 0:
                    values.append(str(None))
                else:
                    if isinstance(val[-1], list):
                        for it in val[-1]:
                            values.append(str(np.mean(it)))
                    else:
                        values.append(str(val[-1]))
            values.append(cur_time)
            f.write(f'{nr},')
            f.write(','.join(values))
            f.write('\n')

    def feat_update(self, sequences):
        model_type_to_use = self.modeltype
        features_for_chain = {}
        self.log.debug(f'Sequences: {sequences}')
        for sequence_index, (sequence, msa) in enumerate(zip(sequences, self.msas), start=1):
            self.log.debug(f"MSA: {msa}")

            single_chain_msas = [msa]
            uniprot_msa = msa

            # Turn the raw data into model features.
            feature_dict = {}
            feature_dict.update(
                pipeline.make_sequence_features(
                    sequence=sequence, description="query", num_res=len(sequence)
                )
            )

            # Force input template
            if self.temp_paths is not None and self.modeltype == "ptm":
                self.log.debug(f'Building templates: {self.temp_paths}')
                msa, temps = self._build_templates(sequence)
                single_chain_msas = [msa]
                uniprot_msa = msa

                feature_dict.update(temps)
            else:
                # Add only empty placeholder features.
                feature_dict.update(
                    notebook_utils.empty_placeholder_template_features(
                        num_templates=0, num_res=len(sequence)
                    )
                )

            feature_dict.update(pipeline.make_msa_features(msas=single_chain_msas))

            # Construct the all_seq features only for heteromers, not homomers.
            if len(set(sequences)) > 1 and model_type_to_use == "multimer":
                valid_feats = msa_pairing.MSA_FEATURES + (
                    "msa_uniprot_accession_identifiers",
                    "msa_species_identifiers",
                )
                all_seq_features = {
                    f"{k}_all_seq": v
                    for k, v in pipeline.make_msa_features([uniprot_msa]).items()
                    if k in valid_feats
                }
                feature_dict.update(all_seq_features)

            features_for_chain[protein.PDB_CHAIN_IDS[sequence_index - 1]] = feature_dict

        if model_type_to_use == "ptm":
            np_example = features_for_chain[protein.PDB_CHAIN_IDS[0]]

        else:
            all_chain_features = {}
            for chain_id, chain_features in features_for_chain.items():
                all_chain_features[
                    chain_id
                ] = pipeline_multimer.convert_monomer_features(chain_features, chain_id)

            all_chain_features = pipeline_multimer.add_assembly_features(
                all_chain_features
            )

            np_example = feature_processing.pair_and_merge(
                all_chain_features=all_chain_features, is_prokaryote=False
            )  # TODO: make setting

            # Pad MSA to avoid zero-sized extra_msa.
            np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=512)

            np_example["target_feat"] = jax.nn.one_hot(np_example["aatype"], 21)

        feat = self.model.process_features(np_example, random_seed=0)

        # Apply masks
        if self.modeltype == "ptm":
            feat["positional_mask"] = feat["atom37_atom_exists"].copy()
            feat["sidechain_mask"] = feat["atom37_atom_exists"].copy()
            feat["positional_mask"][self.positional_mask.astype(bool)] = 0
            feat["sidechain_mask"][self.sidechain_mask.astype(bool)] = 0

        else:
            feat["positional_mask"] = feat["all_atom_mask"].copy()
            feat["sidechain_mask"] = feat["all_atom_mask"].copy()
            feat["positional_mask"][self.positional_mask[0].astype(bool)] = 0
            feat["sidechain_mask"][self.sidechain_mask[0].astype(bool)] = 0

        return feat

    def _preprocess_gt(self, target_prot):

        self.log.debug("Preprocessing groundtruth!")
        gt_dict = {"all_atom_positions": target_prot.atom_positions}
        if self.modeltype == "multimer":
            gt = all_atom_multimer.atom37_to_frames(
                jnp.expand_dims(target_prot.aatype, 0),
                geometry.Vec3Array.from_array(
                    jnp.expand_dims(target_prot.atom_positions, 0)
                ),
                jnp.expand_dims(target_prot.atom_mask, 0),
            )
            gt_dict.update(gt)

        elif self.modeltype == "ptm":
            gt_dict["gt_aatype"] = np.expand_dims(target_prot.aatype, 0)
            gt_dict["all_atom_positions"] = np.expand_dims(
                gt_dict["all_atom_positions"], 0
            )
        if self.clamped is not None:
            self.log.debug(f"Clamping is {self.clamped}")
            gt_dict["use_clamped_fape"] = [self.clamped]

        self.log.debug("Preprocess complete!")
        return gt_dict


class GradientDesign(Design):

    def step(self, model, gt_dict):
        sequences = self._pssm_to_sequence()

        self.loss_track["sequences"].append(sequences)
        self.log.debug(sequences)

        if self.modeltype == "ptm" and len(sequences) > 1:
            sequences = ["".join(sequences)]

        seqsim = calc_seqsim(''.join(self.gt_seq), ''.join(sequences))

        self.loss_track["seq_sim"].append(seqsim)

        feat = self.feat_update(sequences)
        feat.update(gt_dict)
        if self.modeltype == "ptm":
            feat['pseudo_beta'], feat['pseudo_beta_mask'] = modules.pseudo_beta_fn(feat['gt_aatype'],
                                                                                   feat['all_atom_positions'],
                                                                                   feat["atom37_atom_exists"])

        if self.modeltype == "ptm" and len(self.chains) > 1:
            feat["residue_index"] = self._chainbreak(feat)

        self.loss_track["learning_rate"].append(float(self.lr))

        target = {"target_feat": feat["target_feat"]}

        (loss, result), gradient = model(target, feat, self.model_params)

        # Calculate TM-score and RMSD
        highest_confidence_result, ptm, plddt, loss = self._calc_scores(result, feat, loss)
        self._update_pssm(gradient)
        return sequences, loss, highest_confidence_result, feat, plddt, ptm


class MCMCDesign(Design):
    beta_vals = {'fape_loss': 80, 'tm_loss': 350, 'plddt_loss': 650}

    def __init__(
            self,
            datadir,
            random_seed,
            mcmc_muts,
            surf_optim=False,
            output_path: str = None,
            debug: bool = False,
    ):
        super(MCMCDesign, self).__init__(datadir, output_path, debug)
        # CONFIG
        self.random_seed = random_seed
        self.mcmc_muts = mcmc_muts
        self.surf_optim = surf_optim

        self.mode = 'mcmc'

        self.beta = 0

        np.random.seed(random_seed)  # Set random
        self.log.info(
            f"MCMC with the following settings:\n number of mutations: {self.mcmc_muts} \n "
            f"surface optimization: {self.surf_optim} \n seed number: {self.random_seed}"
        )
        self.best_score_mc = np.inf
        self.aa_bkgr = np.array(
            [
                0.07892653,
                0.04979037,
                0.0451488,
                0.0603382,
                0.01261332,
                0.03783883,
                0.06592534,
                0.07122109,
                0.02324815,
                0.05647807,
                0.09311339,
                0.05980368,
                0.02072943,
                0.04145316,
                0.04631926,
                0.06123779,
                0.0547427,
                0.01489194,
                0.03705282,
                0.0691271,
            ],
            dtype=np.float32,
        )

        self.surf_aa_bkgr = np.array(
            [
                0.0,
                0.08333333,
                0.08333333,
                0.08333333,
                0.08333333,
                0.08333333,
                0.08333333,
                0.08333333,
                0.08333333,
                0.0,
                0.0,
                0.08333333,
                0.0,
                0.0,
                0.08333333,
                0.08333333,
                0.08333333,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def _prep_opt(self):
        self.optimizer = None
        self.opt_state = None

    def set_loss(self, loss):
        super(MCMCDesign, self).set_loss(loss)

        for key in self.loss_fn.keys():
            self.beta += self.beta_vals.get(key.lower(), 0)

    def find_residue_localization(self):
        """Takes in a list of the AlphaFold generated pdb, writes it out
        in a temporary folder, and returns the surface residues."""

        # Generate a temporary folder in which to store the pdb file.
        with TemporaryDirectory(prefix="temp") as tmpdir:
            f = open(tmpdir + "/temp_struct.pdb", "w")
            f.write(self.pdbs[-1])
            f.close()

            # load into pyrosetta
            structure = pose_from_pdb(tmpdir + "/temp_struct.pdb")

        # Select the surface residues from the structure.
        layer_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(structure)

        # Select the hydrophobic residues from the structure.
        res_sel = (
            pyrosetta.rosetta.core.select.residue_selector.ResiduePropertySelector()
        )
        res_sel.add_property(
            pyrosetta.rosetta.core.chemical.ResidueProperty.HYDROPHOBIC
        )
        hphobic_res = res_sel.apply(structure)

        surface_resi = []
        hphobic_resi = []
        for i in range(1, len(surface_res) + 1):
            if surface_res[i] is True:
                surface_resi.append(i - 1)
            if hphobic_res[i] is True:
                hphobic_resi.append(i - 1)

        return surface_resi, hphobic_resi

    def step(self, model, gt_dict):

        # save the old pssm and create a new one to be mutated
        prev_pssm = np.array(self.pssm.copy())
        pssm = np.array(self.pssm.copy())

        # Initialize pyrosetta when a structure becomes available.
        if len(self.pdbs) == 1 and self.surf_optim:
            init("-mute all -ignore_unrecognized_res")

        # check if a structure is available. If not, run first without mutations.
        if len(self.pdbs) > 0:
            # introduce random mutations and make sure they are not in the position mask.
            pos_loop = True
            while pos_loop:
                pos = np.random.choice(pssm.shape[0], self.mcmc_muts)
                # this only works for single mutations for now
                for i in pos:
                    if np.float(i) not in self.mcmc_mask:
                        pos_loop = False

            if self.surf_optim:
                # Determine the surface residues of this structure.
                surface_residues, hphobic_residues = self.find_residue_localization()

                # determine the surface hydrophobics.
                surf_hydrophobics = []
                for i in hphobic_residues:
                    if i in surface_residues:
                        surf_hydrophobics.append(i)

                # Force mutations towards the hydrophobic sites.
                if len(surf_hydrophobics) <= self.mcmc_muts:
                    pos = np.random.choice(len(surf_hydrophobics), self.mcmc_muts)

                aa = []
                for i in pos:
                    # Surface residues get a special probability distribution
                    # preventing the sampling of hydrophobics on the surface.
                    if i in surface_residues:
                        aa.append(np.random.choice(20, 1, p=self.surf_aa_bkgr))
                    else:
                        aa.append(np.random.choice(20, 1, p=self.aa_bkgr))
            else:
                aa = np.random.choice(20, self.mcmc_muts, p=self.aa_bkgr)

            pssm[pos, :] = 0
            pssm[pos, aa] = 1

        pssm[self.pssm == -np.inf] = -np.inf
        self.pssm = pssm
        sequences = self._pssm_to_sequence()

        if self.modeltype == "ptm" and len(sequences) > 1:
            sequences = ["".join(sequences)]

        self.loss_track["sequences"].append(sequences)

        self.log.debug(sequences)
        feat = self.feat_update(sequences)

        feat.update(gt_dict)
        if self.modeltype == "ptm":
            feat['pseudo_beta'], feat['pseudo_beta_mask'] = modules.pseudo_beta_fn(feat['gt_aatype'],
                                                                                   feat['all_atom_positions'],
                                                                                   feat["atom37_atom_exists"])

        if self.modeltype == "ptm" and len(self.chains) > 1:
            feat["residue_index"] = self._chainbreak(feat)

        target = {
            "target_feat": feat["target_feat"]
        }

        loss, result = model(target, feat, self.model_params)
        result0 = jax.tree_map(lambda x: x[0], result)
        # Calculate TM-score and RMSD if it is a monomer design.
        if len(self.chains) == 1:
            TM_aligned_mean, RMSD_aligned_mean = extract_metrics(result, feat, self.target_file)
            # Save the results
            result0["structure_module"]["TM_loss"] = TM_aligned_mean
            result0["structure_module"]["RMSD_loss"] = RMSD_aligned_mean

        highest_confidence_result, ptm, plddt, loss = self._calc_scores(result, feat, loss)
        loss = jnp.mean(loss)

        if loss < self.best_score_mc:
            self.best_score_mc = loss
            self.loss = loss
        elif np.exp((self.best_score_mc - loss) * self.beta) > np.random.uniform():
            self.best_score_mc = loss
        else:
            self.pssm = prev_pssm

        if self.counter % 250 == 0 and self.counter != 0:
            self.beta = self.beta * 2 if self.beta < 10000 else 10000  # Limit to avoid overflow

        return sequences, loss, result0, feat, plddt, ptm

    def _prep_model(self):
        # AF FUNCTION
        def alphafold_fn(target, processed_feature_dict, model_params):
            feat2 = {**processed_feature_dict}
            feat2.update(target)
            result, _ = self.model.apply(model_params, jax.random.PRNGKey(0), feat2)
            result["design_loss"] = {}
            overall_loss = 0.0
            for k, l_fn in self.loss_fn.items():
                loss = l_fn(result, feat=feat2, target_file=self.target_file)
                result["design_loss"][k] = loss
                overall_loss = overall_loss + jnp.mean(loss)

            return overall_loss, result

        alphafold_fwd = jax.jit(jax.vmap(alphafold_fn, in_axes=(None, None, 0)))
        return alphafold_fwd
