# -*- coding: utf-8 -*-

import typing
from itertools import chain, combinations

import torch

from mmvae_hub.base.evaluation.divergence_measures.mm_div import calc_alphaJSD_modalities, poe
from mmvae_hub.base.utils import utils


def moe_fusion(flags, mus, logvars, weights):
    weights = utils.reweight_weights(weights)
    mu_moe, logvar_moe = utils.mixture_component_selection(flags, mus, logvars, weights)
    return [mu_moe, logvar_moe]


def poe_fusion(flags, mus, logvars, weights=None):
    """
    Fuses all modalities in subset with product of experts method.
    """
    if flags.method == 'poe':
        num_samples = mus[0].shape[0]
        mus = torch.cat((mus, torch.zeros(1, num_samples, flags.class_dim).to(flags.device)), dim=0)
        logvars = torch.cat((logvars, torch.zeros(1, num_samples, flags.class_dim).to(flags.device)), dim=0)
    mu_poe, logvar_poe = poe(mus, logvars)
    return [mu_poe, logvar_poe]


def fusion_condition_moe(subset, batch_mods) -> bool:
    return len(subset) == 1


def fusion_condition_poe(subset, input_batch=None):
    return len(subset) == len(input_batch.keys())


def fusion_condition_joint(subset, input_batch=None):
    return True


# def divergence_static_prior(flags, mus, logvars, weights):
#     weights = weights.clone()
#     weights = utils.reweight_weights(weights)
#     div_measures = calc_group_divergence_moe(flags, mus, logvars, weights, normalization=flags.batch_size)
#     return {
#         'joint_divergence': div_measures[0],
#         'individual_divs': div_measures[1],
#         'dyn_prior': None,
#     }


def divergence_dynamic_prior(flags, mus, logvars, weights=None):
    div_measures = calc_alphaJSD_modalities(flags,
                                            mus,
                                            logvars,
                                            weights,
                                            normalization=flags.batch_size)
    return {
        'joint_divergence': div_measures[0],
        'individual_divs': div_measures[1],
        'dyn_prior': div_measures[2],
    }


def subsets_from_batchmods(batchmods: typing.Iterable[str]) -> set:
    """
    >>> subsets_from_batchmods(batchmods = ['m0', 'm1', 'm2'])
    {'m0_m2', 'm2', 'm1_m2', 'm0_m1_m2', 'm0_m1', 'm0', 'm1'}
    """
    subsets_list = chain.from_iterable(combinations(batchmods, n) for n in range(len(batchmods) + 1))
    subsets = ['_'.join(sorted(mod_names)) for mod_names in subsets_list if mod_names]
    return set(sorted(subsets))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
