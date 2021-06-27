from abc import abstractmethod

from mmvae_hub.evaluation.divergence_measures.kl_div import calc_entropy_gauss, calc_divergence_embedding
from mmvae_hub.evaluation.divergence_measures.kl_div import calc_kl_divergence, calc_kl_divergence_flow
from mmvae_hub.modalities import BaseModality
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.utils import reweight_weights


class BaseMMDiv:
    def __init__(self):
        self.calc_kl_divergence = calc_kl_divergence

    def calc_modality_divergence(self, m1_mu, m1_logvar, m2_mu, m2_logvar, flags):

        uniform_mu = torch.zeros(m1_mu.shape).to(flags.device)
        uniform_logvar = torch.zeros(m1_logvar.shape).to(flags.device)
        klds = torch.zeros(3, 3).to(flags.device)
        klds_modonly = torch.zeros(2, 2).to(flags.device)

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

    def calc_klds(self, forward_results: BaseForwardResults, subsets: Mapping[str, BaseModality], num_samples: int,
                  joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        latent_subsets = forward_results.joint_latents.subsets
        klds = self.calc_subset_divergences(latent_subsets)

        # joint_div average of all subset divs
        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))
        # normalize with the number of samples
        joint_div = joint_div.mean() * (1 / float(num_samples))

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


class GfMoPDiv(BaseMMDiv):
    """MM Div for generalized f-mean of product of experts methods."""

    def __init__(self):
        super().__init__()
        self.calc_kl_divergence_subsets = calc_kl_divergence
        self.calc_kl_divergence_joint = calc_divergence_embedding

    def calc_klds(self, forward_results: BaseForwardResults, subsets: Mapping[str, BaseModality], num_samples: int,
                  joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        klds = self.calc_subset_divergences(subsets=forward_results.joint_latents.subsets)

        # the joint div is calculated with the negative log probabilities of the embeddings.
        joint_div = self.calc_kl_divergence_joint(forward_results.joint_latents.joint_embedding.embedding)
        # normalize with number of samples
        joint_div = joint_div * (1 / float(num_samples))

        # normalize klds with number of modalities in subset and batch_size
        for subset_key, subset in subsets.items():
            weights = (1 / float(num_samples)) * torch.ones(len(subset)).to(klds[subset_key].device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        return klds, joint_div

    def calc_subset_divergences(self, subsets: Mapping[str, Distr]):
        return {
            s_key: self.calc_kl_divergence_subsets(distr0=distr_subset)
            for s_key, distr_subset in subsets.items()
        }

    def calc_group_divergence(self, device, forward_results: BaseForwardResults, normalization=None) -> BaseDivergences:
        pass


class MoFoPDiv(BaseMMDiv):
    """MM Div for mixture of flow of products of experts methods."""

    def __init__(self):
        super().__init__()
        self.calc_kl_divergence_subsets = calc_kl_divergence_flow
        self.calc_kl_divergence_joint = calc_divergence_embedding

    def calc_subset_divergences(self, subsets: Mapping[str, SubsetFoS]):
        return {
            s_key: self.calc_kl_divergence_subsets(q0=subset.q0, z0=subset.z0, zk=subset.zk, log_det_j=subset.log_det_j)
            for s_key, subset in subsets.items()
        }

    def calc_group_divergence(self, device, forward_results: BaseForwardResults, normalization=None) -> BaseDivergences:
        pass


class GfMMMDiv(BaseMMDiv):
    def __init__(self):
        super().__init__()
        self.calc_kl_divergence_unimodal = calc_kl_divergence
        self.calc_kl_divergence_multimodal = calc_divergence_embedding

    def calc_klds(self, forward_results: BaseForwardResults, subsets: Mapping[str, BaseModality], num_samples: int,
                  joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        latent_subsets = forward_results.joint_latents.subsets
        klds_singlemod = self.calc_singlemod_divergences(enc_mods=forward_results.enc_mods)
        # the multi modal divergences are calculated with the negative log probabilities of the embeddings.
        klds_multimod = self.calc_multimod_divergences(latent_subsets)

        klds = klds_singlemod | klds_multimod

        # normalize klds with number of modalities in subset and batch_size
        for subset_key, subset in subsets.items():
            weights = (1 / float(num_samples)) * torch.ones(len(subset)).to(klds[subset_key].device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        joint_div = klds['_'.join(joint_keys)]
        return klds, joint_div

    def calc_multimod_divergences(self, latent_subsets: Mapping[str, Distr]):
        return {
            mod_str: self.calc_kl_divergence_multimodal(subset)
            for mod_str, subset in latent_subsets.items() if len(mod_str.split('_')) > 1
        }

    def calc_singlemod_divergences(self, enc_mods: Mapping[str, BaseEncMod]):
        return {
            mod_str: self.calc_kl_divergence_unimodal(distr0=enc_mod.latents_class, enc_mod=enc_mod)
            for mod_str, enc_mod in enc_mods.items()
        }

    def calc_group_divergence(self, device, forward_results: BaseForwardResults, normalization=None) -> BaseDivergences:
        pass


class EGfMMMDiv(BaseMMDiv):
    def __init__(self):
        super().__init__()
        self.calc_kl_divergence = calc_divergence_embedding

    def calc_klds(self, forward_results: BaseForwardResults, subsets: Mapping[str, BaseModality], num_samples: int,
                  joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        latent_subsets = forward_results.joint_latents.subsets

        # the divergences are calculated with the negative log probabilities of the embeddings.
        klds = self.calc_divergences(latent_subsets)

        # normalize klds with number of modalities in subset and batch_size
        for subset_key, subset in subsets.items():
            weights = (1 / float(len(subset) * num_samples)) * torch.ones(len(subset)).to(klds[subset_key].device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        joint_div = klds['_'.join(joint_keys)]
        return klds, joint_div

    def calc_group_divergence(self, device, forward_results: BaseForwardResults, normalization=None) -> BaseDivergences:
        pass

    def calc_divergences(self, latent_subsets: Mapping[str, Distr]):
        return {
            mod_str: self.calc_kl_divergence(subset)
            for mod_str, subset in latent_subsets.items()
        }


class PGfMMMDiv(BaseMMDiv):
    def __init__(self):
        super().__init__()

    def calc_klds(self, forward_results: BaseForwardResults, subsets: Mapping[str, BaseModality], num_samples: int,
                  joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        latent_subsets = forward_results.joint_latents.subsets
        klds = self.calc_subset_divergences(latent_subsets)

        joint_div = klds['_'.join(joint_keys)]
        return klds, joint_div

    def calc_group_divergence(self, device, forward_results: BaseForwardResults, normalization=None) -> BaseDivergences:
        pass


class FlowVAEMMDiv(BaseMMDiv):
    """Class of MMDivs for methods that use flows."""

    def __init__(self):
        super().__init__()


class FoJMMDiv(FlowVAEMMDiv):
    """"FlowOfJointMMDiv: Class of MMDivs where for methods where the flow is applied on the joint distribution."""

    def __init__(self):
        super().__init__()


class FoSMMDiv(FlowVAEMMDiv):
    """"FlowOfJointMMDiv: Class of MMDivs where for methods where the flow is applied on each subset."""

    def __init__(self):
        super().__init__()


class FoEncModsMMDiv(FlowVAEMMDiv):
    """"JointFromFlowMMDiv: Class of MMDivs where for methods where the flow is applied on each encoded modality."""

    def __init__(self):
        super().__init__()
        self.calc_kl_divergence = calc_kl_divergence

    def calc_klds(self, forward_results: BaseForwardResults, subsets, num_samples: int, joint_keys: Iterable[str]):
        latents: JointLatentsFoS = forward_results.joint_latents
        klds = self.calc_subset_divergences(latents.subsets)

        for k, kl_q0 in klds.items():
            # ln p(z_k)  (not averaged)
            log_p_zk = calc_divergence_embedding(latents.subsets[k].zk)
            # N E_q0[ ln q(z_0) - ln p(z_k) ]
            diff = kl_q0 - log_p_zk
            # to minimize the divergence,
            summed_logs = diff

            summed_ldj = torch.sum(latents.subsets[k].log_det_j)

            # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
            klds[k] = (summed_logs - summed_ldj)

        # normalize klds with number of modalities in subset and batch_size
        for subset_key, subset in subsets.items():
            weights = 1 / float(num_samples)
            klds[subset_key] = weights * klds[subset_key].squeeze()

        joint_div = klds['_'.join(joint_keys)]

        return klds, joint_div

    def calc_subset_divergences(self, latent_subsets: Mapping[str, SubsetFoS]):
        return {
            mod_str: self.calc_kl_divergence(distr0=subset.q0)
            for mod_str, subset in latent_subsets.items()
        }


class NoFlowVAEMMDiv(BaseMMDiv):
    """Class of MMDivs for methods that do not use flows."""

    def __init__(self):
        super().__init__()


class POEMMDiv(NoFlowVAEMMDiv):
    def __init__(self):
        super().__init__()

    def calc_modality_divergence(self, m1_mu, m1_logvar, m2_mu, m2_logvar, flags):
        return self.calc_kl_divergence(m1_mu, m1_logvar, m2_mu, m2_logvar, norm_value=flags.batch_size).sum()


class MixtureMMDiv(NoFlowVAEMMDiv):
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


class JointElbowMMDiv(MixtureMMDiv, POEMMDiv):
    def __init__(self):
        super().__init__()


class FoMoPMMDiv(FoSMMDiv, JointElbowMMDiv):
    """Planar Flow of Mixture of product of experts multi-modal divergence"""

    def __init__(self):
        FoSMMDiv.__init__(self)
        JointElbowMMDiv.__init__(self)


class PoPEMMDiv(FoEncModsMMDiv, POEMMDiv):
    """Planar Flow of Mixture multi-modal divergence"""

    def __init__(self):
        FoEncModsMMDiv.__init__(self)
        POEMMDiv.__init__(self)


class PfomMMDiv(FoEncModsMMDiv, MixtureMMDiv):
    """Planar Flow of Mixture multi-modal divergence"""

    def __init__(self):
        FoEncModsMMDiv.__init__(self)
        MixtureMMDiv.__init__(self)


class PlanarMixtureMMDiv(FoEncModsMMDiv, MixtureMMDiv):
    def __init__(self):
        super().__init__()

        self.calc_kl_divergence = calc_kl_divergence_flow

    def calc_klds(self, forward_results: BaseForwardResults, subsets, num_samples: int, joint_keys: Iterable[str]):
        """
        Calculate the Kl divergences for all subsets and the joint latent distribution.
        Calculate first the single mod divergences that are then used to compute the subset divergences.
        """

        klds = self.calc_singlemod_divergences(forward_results.enc_mods)
        klds = self.calc_subset_divergences(klds, subsets)

        for subset_key, subset in subsets.items():
            weights = (1 / float(len(subset) * num_samples)) * torch.ones(len(subset)).to(klds[subset_key].device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        joint_div = klds['_'.join(joint_keys)]
        return klds, joint_div

    def calc_subset_divergences(self, klds, subsets):
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
        MixtureMMDiv.__init__(self)
        POEMMDiv.__init__(self)


class FoMFoPMMDiv(FlowVAEMMDiv):
    """"JointFromFlowMMDiv: Class of MMDivs where for methods where the flow is applied on each modality."""

    def __init__(self):
        super().__init__()
        self.calc_kl_divergence = calc_kl_divergence

    def calc_klds(self, forward_results: BaseForwardResults, subsets: Mapping[str, BaseModality], num_samples: int,
                  joint_keys: Iterable[str]):
        """Calculate the Kl divergences for all subsets and the joint latent distribution."""

        latent_subsets = forward_results.joint_latents.subsets
        klds = self.calc_subset_divergences(latent_subsets)

        joint_div = torch.Tensor().to(klds[list(klds)[0]].device)
        for s_key in joint_keys:
            joint_div = torch.cat((joint_div, torch.atleast_1d(klds[s_key])))

        weights = (1 / float(len(joint_keys) * num_samples)) * torch.ones(len(joint_div)).to(joint_div.device)
        joint_div = (weights * joint_div).sum(dim=0) - forward_results.joint_latents.joint_distr.log_det_j

        # normalize klds with number of modalities in subset and batch_size
        for subset_key, subset in subsets.items():
            weights = (1 / float(len(subset) * num_samples)) * torch.ones(len(subset)).to(joint_div.device)
            klds[subset_key] = (weights * klds[subset_key].squeeze()).sum(dim=0)

        return klds, joint_div

    def calc_subset_divergences(self, latent_subsets: Mapping[str, SubsetFoS]):
        return {
            mod_str: self.calc_kl_divergence(distr0=subset.q0) - subset.log_det_j
            for mod_str, subset in latent_subsets.items()
        }

    def calc_klds(self, forward_results: BaseForwardResults, subsets, num_samples: int, joint_keys: Iterable[str]):
        # calculate divergences of the q0 distributions
        klds, joint_div = super().calc_klds(forward_results, subsets, num_samples, joint_keys)

        # joint_div = \sum E_q0[ ln qi(z|X) - ln p(z) ] - E_q_z0[\sum_k log |det dz_k/dz_k-1|].
        joint_div = joint_div - torch.sum(forward_results.joint_latents.joint_embedding.log_det_j)

        for s_key, kld in klds.items():
            klds[s_key] = kld - torch.sum(forward_results.joint_latents.subsets[s_key].log_det_j)

        return klds, joint_div

    def calc_subset_divergences(self, latent_subsets: Mapping[str, SubsetFoS]):
        return {
            mod_str: self.calc_kl_divergence(distr0=subset.q0)
            for mod_str, subset in latent_subsets.items()
        }
