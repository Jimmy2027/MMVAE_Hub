# -*- coding: utf-8 -*-
import torch

from mmvae_hub.base.BaseMMVae import BaseMMVAE
from mmvae_hub.base.utils import utils
from mmvae_hub.base.utils.Dataclasses import Distr
from mmvae_hub.evaluation.divergence_measures.mm_div import MixtureMMDiv, JointElbowMMDiv, JSDMMDiv


class MOEMMVae(BaseMMVAE):
    def __init__(self, exp, flags, modalities, subsets):
        super(MOEMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = MixtureMMDiv()

    @staticmethod
    def modality_fusion(flags, mus, logvars, weights) -> Distr:
        """Fuse modalities with the mixture of experts method."""
        weights = utils.reweight_weights(weights)
        return MOEMMVae.mixture_component_selection(flags, mus, logvars, weights)

    @staticmethod
    def fusion_condition(subset, batch_mods) -> bool:
        return len(subset) == 1

    @staticmethod
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


class JointElboMMVae(MOEMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        super(JointElboMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = JointElbowMMDiv()

    def modality_fusion(self, flags, mus, logvars, weights=None) -> Distr:
        """
        Fuses all modalities in subset with product of experts method.
        """
        mu_poe, logvar_poe = self.mm_div.poe(mus, logvars)
        return Distr(mu_poe, logvar_poe)

    @staticmethod
    def fusion_condition(subset, batch_mods) -> bool:
        return True

    @staticmethod
    def calc_elbo(exp, modality, recs, klds):
        flags = exp.flags
        s_weights = exp.style_weights
        kld_content = klds['content']
        if modality == 'joint':
            w_style_kld = 0.0
            w_rec = 0.0
            klds_style = klds['style']
            mods = exp.modalities
            r_weights = exp.rec_weights
            for m_key in mods:
                w_style_kld += s_weights[m_key] * klds_style[m_key]
                w_rec += r_weights[m_key] * recs[m_key]
            kld_style = w_style_kld
            rec_error = w_rec
        else:
            beta_style_mod = s_weights[modality]
            # rec_weight_mod = r_weights[modality]
            rec_weight_mod = 1.0
            kld_style = beta_style_mod * klds['style'][modality]
            rec_error = rec_weight_mod * recs[modality]
        div = flags.beta_content * kld_content + flags.beta_style * kld_style
        return rec_error + flags.beta * div


class JSDMMVae(JointElboMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        super(JSDMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = JSDMMDiv()
