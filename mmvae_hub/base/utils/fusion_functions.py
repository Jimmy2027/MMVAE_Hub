# -*- coding: utf-8 -*-

import torch

from mmvae_hub.base.evaluation.divergence_measures.mm_div import calc_alphaJSD_modalities, calc_group_divergence_moe, \
    poe
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


def fusion_condition_moe(subset) -> bool:
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
