from abc import abstractmethod
from typing import Iterable, Mapping

import numpy as np
import torch

from mmvae_hub.evaluation.divergence_measures.kl_div import calc_entropy_gauss
from mmvae_hub.evaluation.divergence_measures.kl_div import calc_kl_divergence, calc_kl_divergence_flow
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.utils import reweight_weights


class BaseMMDiv:
    def __init__(self):
        self.calc_kl_divergence = calc_kl_divergence

    def calc_modality_divergence(self, m1_mu, m1_logvar, m2_mu, m2_logvar, flags):

        uniform_mu = torch.zeros(m1_mu.shape)
        uniform_logvar = torch.zeros(m1_logvar.shape)
        klds = torch.zeros(3, 3)
        klds_modonly = torch.zeros(2, 2)
        if flags.cuda:
            klds = klds.cuda()
            klds_modonly = klds_modonly.cuda()
            uniform_mu = uniform_mu.cuda()
            uniform_logvar = uniform_logvar.cuda()

        mus = [uniform_mu, m1_mu, m2_mu]
        logvars = [uniform_logvar, m1_logvar, m2_logvar]
        for i in range(1, len(mus)):  # CAREFUL: index starts from one, not zero
            for j in range(len(mus)):
                kld = self.calc_kl_divergence(mus[i], logvars[i], mus[j], logvars[j], norm_value=flags.batch_size)
                klds[i, j] = kld
                if i >= 1 and j >= 1:
                    klds_modonly[i - 1, j - 1] = kld
        klds = klds.sum() / (len(mus) * (len(mus) - 1))
        klds_modonly = klds_modonly.sum() / ((len(mus) - 1) * (len(mus) - 1))
        return [klds, klds_modonly]

    def calc_klds(self, forward_results: BaseForwardResults, subsets, num_samples: int, joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        latent_subsets = forward_results.joint_latents.subsets
        klds = self.calc_subset_divergences(latent_subsets)

        joint_div = torch.Tensor().to(klds[list(klds)[0]].device)
        for s_key in joint_keys:
            joint_div = torch.cat((joint_div, torch.atleast_1d(klds[s_key])))

        weights = (1 / float(len(joint_keys) * num_samples)) * torch.ones(len(joint_div)).to(joint_div.device)
        joint_div = (weights * joint_div).sum(dim=0)

        # normalize klds with number of modalities in subset and batch_size
        for subset_key, subset in subsets.items():
            weights = (1 / float(len(subset) * num_samples)) * torch.ones(len(subset)).to(joint_div.device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        return klds, joint_div

    def calc_subset_divergences(self, latent_subsets: Mapping[str, Distr]):
        return {
            mod_str: self.calc_kl_divergence(distr0=subset)
            for mod_str, subset in latent_subsets.items()
        }

    def calc_klds_style(self, flags, latents_mods: dict):
        klds = {}
        for key in latents.keys():
            if key.endswith('style'):
                mu, logvar = latents[key]
                klds[key] = calc_kl_divergence(mu, logvar,
                                               norm_value=flags.batch_size)
        return klds

    @abstractmethod
    def calc_group_divergence(self, device, forward_results: BaseForwardResults, normalization=None) -> BaseDivergences:
        pass


class POEMMDiv(BaseMMDiv):
    def __init__(self):
        super().__init__()

    def calc_group_divergence(self, flags, mus, logvars, norm=None):
        num_mods = mus.shape[0]
        poe_mu, poe_logvar = self.poe(mus, logvars)
        kld_poe = self.calc_kl_divergence(poe_mu, poe_logvar, norm_value=norm)
        klds = torch.zeros(num_mods).to(flags.device)
        for k in range(num_mods):
            kld_ind = self.calc_kl_divergence(mus[k, :, :], logvars[k, :, :],
                                              norm_value=norm)
            klds[k] = kld_ind
        return kld_poe, klds, [poe_mu, poe_logvar]

    @staticmethod
    def poe(mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

    def calc_modality_divergence(self, m1_mu, m1_logvar, m2_mu, m2_logvar, flags):
        return self.calc_kl_divergence(m1_mu, m1_logvar, m2_mu, m2_logvar, norm_value=flags.batch_size).sum()


class MixtureMMDiv(BaseMMDiv):
    def __init__(self):
        super().__init__()


class JSDMMDiv(MixtureMMDiv, POEMMDiv):
    def __init__(self):
        super().__init__()

    def calc_alphaJSD_modalities_mixture(self, m1_mu, m1_logvar, m2_mu, m2_logvar, flags):
        klds = torch.zeros(2)
        entropies_mixture = torch.zeros(2)
        w_modalities = torch.Tensor(flags.alpha_modalities[1:])
        if flags.cuda:
            w_modalities = w_modalities.cuda()
            klds = klds.cuda()
            entropies_mixture = entropies_mixture.cuda()
        w_modalities = reweight_weights(w_modalities)

        mus = [m1_mu, m2_mu]
        logvars = [m1_logvar, m2_logvar]
        for k in range(len(mus)):
            ent = calc_entropy_gauss(flags, logvars[k], norm_value=flags.batch_size)
            # print('entropy: ' + str(ent))
            # print('lb: ' )
            kld_lb = self.calc_kl_divergence_lb_gauss_mixture(flags, k, mus[k], logvars[k], mus, logvars,
                                                              norm_value=flags.batch_size)
            print('kld_lb: ' + str(kld_lb))
            # print('ub: ')
            kld_ub = self.calc_kl_divergence_ub_gauss_mixture(flags, k, mus[k], logvars[k], mus, logvars, ent,
                                                              norm_value=flags.batch_size)
            print('kld_ub: ' + str(kld_ub))
            # kld_mean = (kld_lb+kld_ub)/2
            entropies_mixture[k] = ent.clone()
            klds[k] = 0.5 * (kld_lb + kld_ub)
            # klds[k] = kld_ub
        summed_klds = (w_modalities * klds).sum()
        # print('summed klds: ' + str(summed_klds))
        return summed_klds, klds, entropies_mixture

    def calc_group_divergence(self, flags, mus, logvars, weights, normalization=None):
        num_mods = mus.shape[0]
        num_samples = mus.shape[1]
        alpha_mu, alpha_logvar = self.alpha_poe(weights, mus, logvars)
        if normalization is not None:
            klds = torch.zeros(num_mods)
        else:
            klds = torch.zeros(num_mods, num_samples)
        klds = klds.to(flags.device)

        for k in range(num_mods):
            kld = self.calc_kl_divergence(mus[k, :, :], logvars[k, :, :], alpha_mu,
                                          alpha_logvar, norm_value=normalization)
            if normalization is not None:
                klds[k] = kld
            else:
                klds[k, :] = kld
        if normalization is None:
            weights = weights.unsqueeze(1).repeat(1, num_samples)
        group_div = (weights * klds).sum(dim=0)
        return group_div, klds, [alpha_mu, alpha_logvar]

    @staticmethod
    def alpha_poe(alpha, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        if var.dim() == 3:
            alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)
        elif var.dim() == 4:
            alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        T = 1 / var
        pd_var = 1. / torch.sum(alpha_expanded * T, dim=0)
        pd_mu = pd_var * torch.sum(alpha_expanded * mu * T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

    def get_dyn_prior(self, weights, mus, logvars):
        mu_poe, logvar_poe = self.alpha_poe(weights, mus, logvars)
        return [mu_poe, logvar_poe]


class PfomMMDiv(MixtureMMDiv):
    """Planar Flow of Mixture multi-modal divergence"""

    def __init__(self):
        super(PfomMMDiv).__init__()

        self.calc_kl_divergence = calc_kl_divergence

    def calc_klds(self, forward_results: BaseForwardResults, subsets, num_samples: int, joint_keys: Iterable[str]):
        """
        Calculate the Kl divergences for all subsets and the joint latent distribution.
        Calculate first the single mod divergences that are then used to compute the subset divergences.
        """

        klds = self.calc_singlemod_divergences(forward_results.enc_mods)
        klds = self.calc_subset_divergence(klds, subsets)

        joint_div = torch.Tensor().to(klds[list(klds)[0]].device)
        for s_key in joint_keys:
            joint_div = torch.cat((joint_div, klds[s_key]))

        weights = (1 / float(len(joint_keys) * num_samples)) * torch.ones(len(joint_div)).to(joint_div.device)

        # joint_div = \sum E_q0[ ln qi(z|X) - ln p(z) ] - E_q_z0[\sum_k log |det dz_k/dz_k-1|].
        joint_div = (weights * joint_div).sum(dim=0) - torch.sum(
            forward_results.joint_latents.joint_embedding.log_det_j)

        assert not np.isnan(joint_div.cpu().item())

        for subset_key, subset in subsets.items():
            weights = (1 / float(len(subset) * num_samples)) * torch.ones(len(subset)).to(joint_div.device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0) - torch.sum(
                forward_results.joint_latents.subsets[subset_key].log_det_j)

        return klds, joint_div

    def calc_subset_divergence(self, klds: dict, subsets) -> dict:
        """Concatenate all singlemod klds for all subsets."""
        for subset_key, subset in subsets.items():
            kld = torch.Tensor().to(klds[list(klds)[0]].device)
            for mod in subset:
                kld = torch.cat((kld, torch.atleast_1d(klds[mod.name])))
            klds[subset_key] = kld
        return klds

    def calc_singlemod_divergences(self, enc_mods: Mapping[str, BaseEncMod]):
        return {
            mod_str: self.calc_kl_divergence(
                distr0=enc_mod.latents_class
            )
            for mod_str, enc_mod in enc_mods.items()
        }


class PlanarMixtureMMDiv(MixtureMMDiv):
    def __init__(self):
        super().__init__()

        self.calc_kl_divergence = calc_kl_divergence_flow

    def calc_klds(self, forward_results: BaseForwardResults, subsets, num_samples: int, joint_keys: Iterable[str]):
        """
        Calculate the Kl divergences for all subsets and the joint latent distribution.
        Calculate first the single mod divergences that are then used to compute the subset divergences.
        """

        klds = self.calc_singlemod_divergences(forward_results.enc_mods)
        klds = self.calc_subset_divergence(klds, subsets)

        joint_div = torch.Tensor().to(klds[list(klds)[0]].device)
        for s_key in joint_keys:
            joint_div = torch.cat((joint_div, klds[s_key]))

        weights = (1 / float(len(joint_keys) * num_samples)) * torch.ones(len(joint_div)).to(joint_div.device)
        joint_div = (weights * joint_div).sum(dim=0)

        assert not np.isnan(joint_div.cpu().item())

        for subset_key, subset in subsets.items():
            weights = (1 / float(len(subset) * num_samples)) * torch.ones(len(subset)).to(joint_div.device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        return klds, joint_div

    def calc_subset_divergence(self, klds, subsets):
        for subset_key, subset in subsets.items():
            kld = torch.Tensor().to(klds[list(klds)[0]].device)
            for mod in subset:
                kld = torch.cat((kld, torch.atleast_1d(klds[mod.name])))
            klds[subset_key] = kld
        return klds

    def calc_singlemod_divergences(self, enc_mods: Mapping[str, BaseEncMod]):
        return {
            mod_str: self.calc_kl_divergence(
                distr0=enc_mod.latents_class, enc_mod=enc_mod
            )
            for mod_str, enc_mod in enc_mods.items()
        }


class JointElbowMMDiv(MixtureMMDiv, POEMMDiv):
    def __init__(self):
        super().__init__()