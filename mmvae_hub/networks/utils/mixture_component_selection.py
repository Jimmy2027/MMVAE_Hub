# -*- coding: utf-8 -*-

from mmvae_hub.utils.Dataclasses.Dataclasses import *


def mixture_component_selection(flags, mus, logvars, w_modalities=None) -> Distr:
    """
    For every sample, select one of the experts. Return the joint distribution as mixture of experts.

    Every experts gets selected with probability proportional to the corresponding w_modality, this is equivalent to
    taking an expert for a proportion of the batch that is equal to w_modality —which is what is done here.
    This simulates sampling from the sum of experts, which would not be Gaussian.

    mus Tensor: mus of the experts. Has shape (num_experts, bs, class_dim)
    logvars Tensor: logvars of the experts. Has shape (num_experts, bs, class_dim)
    """
    num_components = mus.shape[0]
    # num_samples is the batch_size
    num_samples = mus.shape[1]
    # if not defined, take pre-defined weights
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device)

    idx_start = []
    idx_end = []
    for k in range(num_components):
        i_start = 0 if k == 0 else int(idx_end[k - 1])
        if k == w_modalities.shape[0] - 1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)

    idx_end[-1] = num_samples

    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])

    return Distr(mu=mu_sel, logvar=logvar_sel)
