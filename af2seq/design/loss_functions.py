'''Loss functions used for design'''
import jax
import jax.numpy as jnp
from functools import wraps

from af2seq.design.utils import extract_metrics

LOSSES = {}


def register_loss(fn):
    @wraps(fn)
    def _wrap_model(fn):
        LOSSES[fn.__name__.lower()] = fn
        return fn

    return _wrap_model(fn)


def _calculate_bin_centers(breaks: jnp.ndarray):

    step = breaks[1] - breaks[0]

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = jnp.concatenate(
        [bin_centers, jnp.asarray([bin_centers[-1] + step])], axis=0
    )
    return bin_centers


def predicted_tm_score(
    logits: jnp.ndarray,
    breaks: jnp.ndarray,
    residue_weights=None,
    asym_id=None,
    interface: bool = False,
) -> jnp.ndarray:

    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = jnp.ones(logits.shape[0])

    bin_centers = _calculate_bin_centers(breaks)

    num_res = logits.shape[0]  # (jnp.sum(residue_weights)).astype(int)
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = jnp.max(jnp.asarray([num_res, 19]))

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.

    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    pair_mask = jnp.ones(shape=[num_res, num_res], dtype=bool)

    if interface:
        pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return jnp.asarray(per_alignment[(per_alignment * residue_weights).argmax()])


@register_loss
def fape_loss(result, weight=1,*args,**kwargs):
    combined_loss = (
        result["structure_module"]["fape"]
        + result["structure_module"]["sidechain_fape"]
    )
    return combined_loss * weight


@register_loss
def ptm_loss(
    result, weight=1,*args,**kwargs
):  # Im not sure if the masks are working for other losses except fape. probably not
    result = 1 - predicted_tm_score(
        logits=result["predicted_aligned_error"]["logits"],
        breaks=result["predicted_aligned_error"]["breaks"],
        residue_weights=jnp.ones(result["predicted_aligned_error"]["logits"].shape[0]),
        asym_id=None,
    )
    return result * weight


@register_loss
def tm_loss(result,feat,target_file,weight=1,*args,**kwargs):
    TM,_ = extract_metrics(result, feat,target_file)

    return (1-TM) * weight


@register_loss
def plddt_loss(result, weight=1,*args,**kwargs):
    logits = result["predicted_lddt"]["logits"]
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = jnp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
    probs = jax.nn.softmax(logits)

    plddt = 1 - jnp.mean(jnp.sum(probs * bin_centers[None, :], axis=-1))
    return plddt * weight

@register_loss
def distogram_loss(result, weight=1):
    loss = result['distogram']['loss']
    return loss * weight


@register_loss
def hotspot_loss(result, hotspot_res=(10,20), weight=1):
    #
    # get the coordinates of the hotspot residues
    pos1 = result["structure_module"]["final_atom_positions"][hotspot_res[0]][1]
    pos2 = result["structure_module"]["final_atom_positions"][hotspot_res[1]][1]

    # calculate the euclidean distance between the C-alpha atoms in the two residues in hotspot_res
    distance = jnp.linalg.norm(pos1 - pos2)
    print(result["structure_module"]["final_atom_positions"].shape)
    # print('in the hotspot loss function')

    return distance * weight