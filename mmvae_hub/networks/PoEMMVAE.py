import torch

from mmvae_hub.evaluation.divergence_measures.mm_div import POEMMDiv
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.utils.Dataclasses import Distr


class POEMMVae(BaseMMVAE):
    def __init__(self, exp, flags, modalities, subsets):
        super(POEMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = POEMMDiv()

    def modality_fusion(self, flags, mus, logvars, weights=None) -> Distr:
        """
        Fuses all modalities in subset with product of experts method.
        """
        # question why is this needed? https://github.com/thomassutter/joint_elbo/blob/051875769351a8a9f2779712f33cc4612669f322/utils/BaseMMVae.py#L123
        num_samples = mus[0].shape[0]
        mus = torch.cat((mus, torch.zeros(1, num_samples, flags.class_dim).to(flags.device)), dim=0)
        logvars = torch.cat((logvars, torch.zeros(1, num_samples, flags.class_dim).to(flags.device)), dim=0)
        mu_poe, logvar_poe = POEMMVae.poe(mus, logvars)
        return Distr(mu_poe, logvar_poe)

    @staticmethod
    def fusion_condition(subset, input_batch=None):
        return len(subset) == len(input_batch)

    @staticmethod
    def poe(mu, logvar, eps=1e-8):
        """
        The product of Gaussian experts is itself Gaussian with mean µ = (∑_i µ_i T_i)(∑_i T_i)^-1 and
        covariance V = (∑_i T_i)^-1 where T_i = (V_i)^-1 is the inverse of the covariance. (see Wu and Goodmann 2018)
        """
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        V = torch.sum(T, dim=0)
        pd_var = 1. / V
        pd_mu = torch.sum(mu * T, dim=0) / V
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar