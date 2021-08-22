# -*- coding: utf-8 -*-

from mmvae_hub.evaluation.divergence_measures.mm_div import MixtureMMDiv, JointElbowMMDiv, JSDMMDiv
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.PoEMMVAE import POEMMVae
from mmvae_hub.networks.utils.mixture_component_selection import mixture_component_selection
from mmvae_hub.utils import utils
from mmvae_hub.utils.dataclasses.Dataclasses import *


class MOEMMVae(BaseMMVAE):
    def __init__(self, exp, flags, modalities, subsets):
        super(MOEMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = MixtureMMDiv()

    @staticmethod
    def modality_fusion(flags, mus, logvars, weights) -> Distr:
        """Fuse modalities with the mixture of experts method."""
        weights = utils.reweight_weights(weights)
        return mixture_component_selection(flags, mus, logvars, weights)

    @staticmethod
    def fusion_condition(subset, batch_mods) -> bool:
        return len(subset) == 1


class MoPoEMMVae(MOEMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        super(MoPoEMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = JointElbowMMDiv()

    def modality_fusion(self, flags, mus, logvars, weights=None) -> Distr:
        """
        Fuses all modalities in subset with product of experts method.
        """

        mu_poe, logvar_poe = POEMMVae.poe(mus, logvars)
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


class JSDMMVae(MoPoEMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        super(JSDMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = JSDMMDiv()
