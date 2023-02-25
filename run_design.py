import os
import sys
import argparse
import json
import numpy as np
import jax.numpy as jnp
from af2seq.design.utils import cleanup_for_json

os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"
from af2seq import GradientDesign, MCMCDesign


def main(ARGS):
    os.makedirs(ARGS.out, exist_ok=True)

    if ARGS.mode == "gd":
        d = GradientDesign(ARGS.datadir,ARGS.out, debug=False)
    elif ARGS.mode == "mcmc":
        d = MCMCDesign(
            ARGS.datadir,
            ARGS.seed,
            ARGS.mcmc_muts,
            ARGS.surf_optim,
            ARGS.out,
            debug=False,
        )
    else:
        raise ValueError(f'{ARGS.mode}')
    
    # construct MSA input
    msa_input = []
    if ARGS.msas is None:
        msa_input = None
    else:
        for i in ARGS.msas:
            if i == 'None':
                msa_input.append(None)
            else:
                msa_input.append(i)

    # construct loss dictionary
    loss_fn = {}
    for i in range(0, len(ARGS.loss)):
        loss_fn[ARGS.loss[i]] = float(ARGS.loss_weights[i])

    raise ValueError
    d.design(
        target_file=ARGS.target,
        start_seq=ARGS.startseq,
        msas=msa_input,
        chains=ARGS.chains,
        filename=ARGS.name,
        iterations=ARGS.iter,
        lr=ARGS.lr,
        recycles=ARGS.recycles,
        clampval=ARGS.clamp,
        aa_mask=ARGS.aa_mask,
        fix_pos=ARGS.fix_pos,
        disable_loss_pos=ARGS.disable_loss_pos,
        enable_sc_loss=ARGS.enable_sc_loss,
        modeltype=ARGS.model,
        loss=loss_fn
    )

    # for legacy reasons
    saving = d.loss_track.copy()
    
    saving = cleanup_for_json(saving)
    
    fname = (
        d.name
        if d.name is not None
        else f"{d.target_file.split('/')[-1].split('.pdb')[0]}"
    )

    with open(
        os.path.join(d.output_path, f"design_{fname}_{d.ident}_extended.json"), "w"
    ) as f:
        json.dump(saving, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Af2Seq',description='Fixed backbone design using AlphaFold ')
    # TODO UPDATE the arguments and write help!!!
    parser.add_argument('datadir', help="path to the directory that contains the Alphafold weights")
    parser.add_argument('target', help="target pdb file that is used as groundtruth")
    parser.add_argument("mode", help="Gradient descent (gd) or MCMC (mcmc)", type=str)
    parser.add_argument("out", help="path to output directory")
    parser.add_argument('-n',"--name", help="Name of the experiment", type=str)
    parser.add_argument('-m',"--model", help="Select a specifiy model. ptm or multimer", type=str, default=None)
    parser.add_argument('-c',"--chains",
                        help="chains that are targeted for design. ", nargs="+", type=int, default=None)
    parser.add_argument('-it',"--iter", help="How many design steps should be performed", type=int,default=500)
    parser.add_argument('-s',"--seed", help="seed for mcmc", type=int, default=0)
    parser.add_argument("--lr","--learning_rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument(
        '-l',"--loss",
        help="loss function that is used for the optimization process",
        type=str,
        nargs="+",
        default=["FAPE"],
    )
    parser.add_argument(
        '-lw',"--loss_weights",
        help="specifies the impact of each loss term",
        type=int,
        nargs="+",
        default=[1],
    )
    parser.add_argument('-r',"--recycles", help="AF recycles", type=int, default=0)
    parser.add_argument('-cl',"--clamp", help="FAPE loss clamp clips the loss of the distance between two residues "
                                        "is greater than 10A", type=float, default=0.0)
    parser.add_argument(
        '-am',"--aa_mask",
        help="which amino acids to mask",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        '-fp',"--fix_pos", help="which indexes to mask", nargs="+", type=int, default=None
    )
    parser.add_argument(
        '-dlp',"--disable_loss_pos",
        help="disable backbone FAPE for these positions",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        '-esl',"--enable_sc_loss",
        help="which positions we want use sidechain FAPE in the loss",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        '-st',"--startseq",
        nargs="+",
        help="startseq. A for helix,V for b-sheet G for unordered",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--msas",
        nargs="+",
        help="MSA input path, None for no MSA",
        type=str,
        default=None,
    )
    parser.add_argument(
        '-mm',"--mcmc_muts",
        help="number of mutations introduced each MCMC round",
        type=int,
        default=1,
    )
    parser.add_argument(
        '-so',"--surf_optim",
        help="dont allow hydrophobic mutations on the surface",
        type=bool,
        default=False,
    )

    ARGS = parser.parse_args()
    if len(ARGS.loss) != len(ARGS.loss_weights):
        parser.error(f'loss and loss weights must be of the same length! But loss was: {len(ARGS.loss)} and'
                     f'loss weights: {len(ARGS.loss_weights)}')
    main(ARGS)
