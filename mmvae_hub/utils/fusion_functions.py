# -*- coding: utf-8 -*-

import typing
from itertools import chain, combinations
from typing import Mapping, List

import torch

from mmvae_hub.modalities import BaseModality
from mmvae_hub.utils.Dataclasses import EncModPlanarMixture, Distr, BaseEncMod
from mmvae_hub.utils.utils import split_int_to_bins


def subsets_from_batchmods(batchmods: typing.Iterable[str]) -> set:
    """
    >>> subsets_from_batchmods(batchmods = ['m0', 'm1', 'm2'])
    {'m0_m2', 'm2', 'm1_m2', 'm0_m1_m2', 'm0_m1', 'm0', 'm1'}
    """
    subsets_list = chain.from_iterable(combinations(batchmods, n) for n in range(len(batchmods) + 1))
    subsets = ['_'.join(sorted(mod_names)) for mod_names in subsets_list if mod_names]
    return set(sorted(subsets))


def mixture_component_selection_embedding(enc_mods: typing.Mapping[str, EncModPlanarMixture], s_key: str, flags,
                                          weight_joint: bool = True) -> Distr:
    """
    For each element in batch select an expert from subset.
    s_key: keys of the experts that can be selected. If all experts can be selected (e.g. for MoPoE) s_key should be set to "all".
    """
    num_samples = enc_mods[list(enc_mods)[0]].zk.shape[0]
    s_keys = [s_key for s_key in enc_mods] if s_key == 'all' else s_key.split('_')
    zk_subset = torch.Tensor().to(flags.device)

    if flags.weighted_mixture:
        # define confidence of expert by the mean of zk. Sample experts with probability proportional to confidence.
        confidences = [enc_mods[s_k].zk.mean().abs() for s_k in s_keys]
        bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
        if sum(bins) != num_samples:
            bins[confidences.index(max(confidences))] += num_samples - sum(bins)
    else:
        # fill zk_subset with an equal amount of each expert
        bins = split_int_to_bins(number=num_samples, nbr_bins=len(s_keys))

    # get enc_mods for subset
    for chunk_size, s_k in zip(bins, s_keys):
        zk_subset = torch.cat(
            (zk_subset, enc_mods[s_k].zk[zk_subset.shape[0]:zk_subset.shape[0] + chunk_size]), dim=0)

    assert zk_subset.shape == torch.Size([num_samples, flags.class_dim])

    if weight_joint:
        # normalize latents by number of modalities in subset
        weights_subset = ((1 / float(len(s_keys))) * torch.ones_like(zk_subset).to(flags.device))

        return (weights_subset * zk_subset)
    else:
        return zk_subset


def mixture_component_selection(enc_mods: Mapping[str, BaseEncMod], s_key: str, flags,
                                subsets: Mapping[str, List[BaseModality]]) -> Distr:
    """For each element in batch select an expert from subset."""
    num_samples = enc_mods[list(enc_mods)[0]].latents_class.mu.shape[0]
    mods = subsets[s_key]

    mu_subset = torch.Tensor().to(flags.device)
    logvar_subset = torch.Tensor().to(flags.device)

    if flags.weighted_mixture:
        # define confidence of expert by the inverse of the variance.
        # Sample experts with probability proportional to confidence.
        confidences = [1 / enc_mods[mod.name].latents_class.logvar.exp().mean().abs() for mod in mods]
        bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
        if sum(bins) != num_samples:
            bins[confidences.index(max(confidences))] += num_samples - sum(bins)
    else:
        # fill zk_subset with an equal amount of each expert
        bins = split_int_to_bins(number=num_samples, nbr_bins=len(mods))

    # get enc_mods for subset
    for chunk_size, mod in zip(bins, mods):
        mu_subset = torch.cat((mu_subset, enc_mods[mod.name].latents_class.
                               mu[mu_subset.shape[0]:mu_subset.shape[0] + chunk_size]), dim=0)
        logvar_subset = torch.cat((logvar_subset, enc_mods[mod.name].latents_class.
                                   mu[logvar_subset.shape[0]:logvar_subset.shape[0] + chunk_size]), dim=0)

    assert mu_subset.shape == torch.Size([num_samples, flags.class_dim])

    return Distr(mu_subset, logvar=logvar_subset)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
