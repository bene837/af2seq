import numpy as np

from ..alphafold.data import parsers

from .utils import decode, encode


def get_masking(pssm, aas: list):
    aa_mask = np.zeros_like(pssm)
    for aa in aas:
        aa_mask[:, np.argmax(encode(aa))] = -np.inf

    return aa_mask


def chain_info(target_prot, design_chains):
    report = []
    gt_seq = []

    for index in set(target_prot.chain_index):
        seq = decode(target_prot.aatype[target_prot.chain_index == index])
        gt_seq.append(seq)
        if index in design_chains:
            fixed = "DESIGN"

        else:
            fixed = "FIXED"
        report.append(
            f'[{fixed}] Chain {index}: {f"{seq[:64]}..." if len(seq) > 64 else seq}\n'
        )

    return gt_seq, "\n" + "\n".join(report)


def load_msa(msa_path):
    read = False
    lines = []
    with open(msa_path, "r") as f:
        for l in f:
            if read:
                lines.append(l)
                read = False

            if l.startswith(">"):
                lines.append(l)
                read = True

    a3m = "".join(lines)
    return parsers.parse_a3m(a3m), lines


def build_msa(logger, sequences, msas):
    msa_list = []

    if msas is None:
        msas = [None for s in sequences]

    assert len(sequences) == len(msas), \
        f"Provided {len(sequences)} sequences but msa information for {len(msas)}! Pass None if no msas are used or specifiy per sequence. e.g [msa1,None]"
    for nr, (s, path) in enumerate(zip(sequences, msas)):
        if path is None:
            logger.info(f"No msa was provided for chain {nr}")
            msa_list.append(parsers.Msa([s], [[0] * len(s)], ["query"]))
        else:
            logger.info(f"Found msa for chain {nr}")
            compiled_msa, _ = load_msa(path)
            msa_list.append(compiled_msa)
    return msa_list


def build_msa_ptm(msa_paths, chain_length, sequences):
    # find out the msa depth
    max_depths = []
    for i in msa_paths:
        if i != None:
            _, msa = load_msa(i)
            max_depths.append(len(msa))

    max_msa_depth = min(max_depths)

    new_msa = [""] * max_msa_depth

    for i in range(len(msa_paths)):
        msa_path = msa_paths[i]

        if msa_path is None:
            # go through the max_msa_depth
            for j in range(1, max_msa_depth, 2):
                new_msa[j] = new_msa[j] + "-" * chain_length[i]
        else:
            _, msa = load_msa(msa_path)
            for j in range(max_msa_depth):
                if not msa[j].startswith(">"):
                    new_msa[j] = new_msa[j] + msa[j].replace('\n', '')
                else:
                    new_msa[j] = msa[j]

    # add a copy of the query sequences
    for i in range(len(sequences)):
        seq = sequences.copy()
        seq[i] = len(seq[i]) * '-'
        new_msa.insert(2, ''.join(seq))
        new_msa.insert(2, '>query_seq{nr}\n'.format(nr=i))

    for i in range(1, len(new_msa), 2):
        new_msa[i] = new_msa[i] + "\n"

    a3m = "".join(new_msa)

    new_msa = parsers.parse_a3m(a3m)

    return new_msa


def generate_mcmc_mask(target_prot, chains, fix_pos):
    """Generates a mask for the pssm to fix positions in mcmc mode."""
    mcmc_mask = target_prot.residue_index

    for c in chains:  # chain mask
        mcmc_mask = np.setdiff1d(
            mcmc_mask, target_prot.residue_index[target_prot.chain_index == c]
        )
    if fix_pos != None:
        mcmc_mask = np.append(mcmc_mask, fix_pos)

    # zero indexing
    mcmc_mask = np.unique(mcmc_mask) - 1

    return mcmc_mask


def build_pssm(
        logger,
        target_prot,
        start_seq: list,
        chains: list = None,
        aa_mask: list = None,
        fix_pos: list = None,
        disable_loss_pos: list = None,
        enable_sidechain_loss: list = None,
        encode_value: float = 1.0,
        mode: str = "gd",
):
    """

    Args:


        target_prot: pdb file that contains the target coordinates
        start_seq: list of starting sequences. If None will be random init
        chains: (optional) chains that should be used for design
        aa_mask: (optional) aminoacids which will be ignored during design
        fix_pos: (optional) positions that will not be changed during design,
                             but will not stop the loss from being calculated
        encode_value: (optional) defines the value used to initialize the pssm
        disable_loss_pos: (optional) disables the loss calculation at a specified position
        enable_sidechain_loss: (optional) enables the loss on the sidechains at a position (WIP)


    Returns: starting sequence, the pssm and the corresponding mask

    """

    if isinstance(chains, int):
        chains = [chains]

    gtchains, gtlength = np.unique(target_prot.chain_index, return_counts=True)

    logger.info(f"Found {len(gtchains)} chain(s) of length {gtlength.tolist()}")
    pssm = np.zeros((gtlength.sum(), 20))
    chainmask = np.zeros_like(pssm)

    if chains is None:
        chains = gtchains

    gt_seq, info = chain_info(target_prot, chains)
    logger.info(info)
    if start_seq is None:
        start_seq = []
        logger.warn("WARNING: random init!\nThis might lead to significantly worse results!")
        for s in gtlength:
            start_seq.append(decode(np.random.randint(0, 20, s)))

    if isinstance(start_seq, str):
        start_seq = [start_seq]
    joined_seq = "".join(start_seq)

    pssm = encode(joined_seq)

    # Mask for mcmc round, contains all residues that are to be masked.
    if mode == "mcmc":
        mcmc_mask = generate_mcmc_mask(target_prot, chains, fix_pos)
    else:
        mcmc_mask = []

    for c in chains:  # chain mask
        chainmask[target_prot.chain_index == c] = 1

    if fix_pos is not None:
        for pos in fix_pos:  # position mask
            if pos == 0:
                logger.warn(
                    "zero indexing found! Position mask does not use zero indexing!"
                )

            pssm[pos - 1] = -np.inf
            pssm[pos - 1, encode(joined_seq[pos - 1]).argmax()] = 1

    if aa_mask is not None:
        aa_block = get_masking(pssm, aa_mask)
        pssm[chainmask.astype(bool)] += aa_block[chainmask.astype(bool)]

    pssm[pssm == 1] = encode_value

    positional_mask = np.zeros((1, pssm.shape[0], 37))
    sidechain_mask = np.ones((1, pssm.shape[0], 37))

    positional_mask[0, :, 3:] = 1
    if disable_loss_pos is not None:
        # Check if its not zero indexed.
        for pos in disable_loss_pos:
            if pos == 0:
                logger.warn(
                    "zero indexing found! disable_loss_pos does not use zero indexing!"
                )
        disable_loss_pos = [d - 1 for d in disable_loss_pos]  # zero index
        positional_mask[0, disable_loss_pos] = 1

    if enable_sidechain_loss is not None:
        for pos in enable_sidechain_loss:
            if pos == 0:
                logger.warn(
                    "WARNING: zero indexing found! disable_loss_pos does not use zero indexing!"
                )
        enable_sidechain_loss = [d - 1 for d in enable_sidechain_loss]  # zero index
        sidechain_mask[0, enable_sidechain_loss, :] = 0

    chainmask = chainmask != 0

    return (
        start_seq,
        gt_seq,
        pssm,
        chainmask,
        positional_mask,
        sidechain_mask,
        gtchains,
        gtlength,
        mcmc_mask
    )
