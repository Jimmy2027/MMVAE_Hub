# -*- coding: utf-8 -*-

import typing
from itertools import chain, combinations
from typing import Mapping

import torch
from torch import Tensor

from mmvae_hub.utils.dataclasses.Dataclasses import Distr
from mmvae_hub.utils.utils import split_int_to_bins


def subsets_from_batchmods(batchmods: typing.Iterable[str]) -> set:
    """
    >>> subsets_from_batchmods(batchmods = ['m0', 'm1', 'm2'])
    {'m0_m2', 'm2', 'm1_m2', 'm0_m1_m2', 'm0_m1', 'm0', 'm1'}
    """
    subsets_list = chain.from_iterable(combinations(batchmods, n) for n in range(len(batchmods) + 1))
    subsets = ['_'.join(sorted(mod_names)) for mod_names in subsets_list if mod_names]
    return set(sorted(subsets))


def mixture_component_selection_embedding(subset_embeds: typing.Mapping[str, Tensor], s_key: str, flags,
                                          weight_joint: bool = True) -> Tensor:
    z_subset = torch.Tensor().to(flags.device)
    num_samples = subset_embeds[list(subset_embeds)[0]].shape[0]
    s_keys = [s_key for s_key in subset_embeds] if s_key == 'all' else s_key.split('_')
    experts = torch.cat(tuple(v.unsqueeze(0) for _, v in subset_embeds.items()))

    for sample_idx in range(num_samples):
        z_subset = torch.cat((z_subset, experts[torch.randint(0, len(s_keys), (1,)).item(), sample_idx].unsqueeze(0)))

    return z_subset


def mixture_component_selection_embedding_(subset_embeds: typing.Mapping[str, Tensor], s_key: str, flags,
                                           weight_joint: bool = True) -> Tensor:
    """
    For each element in batch select an expert from subset.
    subset_embeds: embeddings of each subset.
    s_key: keys of the experts that can be selected. If all experts can be selected (e.g. for MoPoE) s_key should be set to "all".
    """
    num_samples = subset_embeds[list(subset_embeds)[0]].shape[0]
    s_keys = [s_key for s_key in subset_embeds] if s_key == 'all' else s_key.split('_')
    z_subset = torch.Tensor(device=flags.device)

    if flags.weighted_mixture:
        # define confidence of expert by the mean of z_subset. Sample experts with probability proportional to confidence.
        confidences = [subset_embeds[s_k].mean().abs() for s_k in s_keys]
        bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
        if sum(bins) != num_samples:
            bins[confidences.index(max(confidences))] += num_samples - sum(bins)
    else:
        # fill zk_subset with an equal amount of each expert
        bins = split_int_to_bins(number=num_samples, nbr_bins=len(s_keys))

    # get enc_mods for subset
    for chunk_size, s_k in zip(bins, s_keys):
        z_subset = torch.cat(
            (z_subset, subset_embeds[s_k][z_subset.shape[0]:z_subset.shape[0] + chunk_size]), dim=0)

    assert z_subset.shape == torch.Size([num_samples, flags.class_dim])

    if weight_joint:
        # normalize latents by number of modalities in subset
        weights_subset = ((1 / float(len(s_keys))) * torch.ones_like(z_subset, device=flags.device))

        return (weights_subset * z_subset)
    else:
        return z_subset


def mixture_component_selection(distrs: Mapping[str, Distr], s_key: str, flags) -> Distr:
    """For each element in batch select an expert from subset."""
    num_samples = distrs[list(distrs)[0]].mu.shape[0]
    s_keys = [s_key for s_key in distrs] if s_key == 'all' else s_key.split('_')

    mu_subset = torch.Tensor().to(flags.device)
    logvar_subset = torch.Tensor().to(flags.device)

    if flags.weighted_mixture:
        # define confidence of expert by the inverse of the variance.
        # Sample experts with probability proportional to confidence.
        confidences = [1 / distrs[mod].logvar.exp().mean().abs() for mod in s_keys]
        bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
        if sum(bins) != num_samples:
            bins[confidences.index(max(confidences))] += num_samples - sum(bins)
    else:
        # fill zk_subset with an equal amount of each expert
        bins = split_int_to_bins(number=num_samples, nbr_bins=len(s_keys))

    # get enc_mods for subset
    for chunk_size, mod in zip(bins, s_keys):
        mu_subset = torch.cat((mu_subset, distrs[mod].
                               mu[mu_subset.shape[0]:mu_subset.shape[0] + chunk_size]), dim=0)
        logvar_subset = torch.cat((logvar_subset, distrs[mod].
                                   mu[logvar_subset.shape[0]:logvar_subset.shape[0] + chunk_size]), dim=0)

    assert mu_subset.shape == torch.Size([num_samples, flags.class_dim])

    return Distr(mu_subset, logvar=logvar_subset)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
