# -*- coding: utf-8 -*-
import typing

import torch
from torch import nn as nn

from mmvae_hub.evaluation.divergence_measures.mm_div import PlanarMixtureMMDiv, PfomMMDiv
from mmvae_hub.networks.MixtureVaes import MOEMMVae
from mmvae_hub.networks.utils import flows
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods
from mmvae_hub.utils.utils import split_int_to_bins


class PlanarFlowMMVAE(MOEMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        super(PlanarFlowMMVAE, self).__init__(exp, flags, modalities, subsets)

        self.mm_div = None

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = flags.num_flows

        # Amortized flow parameters
        if self.num_flows:
            self.amor_u = nn.Linear(flags.class_dim, self.num_flows * flags.class_dim)
            self.amor_w = nn.Linear(flags.class_dim, self.num_flows * flags.class_dim)
            self.amor_b = nn.Linear(flags.class_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def apply_flow(self, in_distr: Distr, flow_params: PlanarFlowParams):
        log_det_j = torch.zeros(in_distr.mu.shape[0]).to(self.flags.device)

        # Sample z_0
        z = [in_distr.reparameterize()]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            # z' = z + u h( w^T z + b)
            z_k, log_det_jacobian = flow_k(z[k], flow_params.u[:, k, :, :], flow_params.w[:, k, :, :],
                                           flow_params.b[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian

        return z[0], z[-1], log_det_j


class PfomMMVAE(PlanarFlowMMVAE):
    """Planar Flow of Mixture multi-modal VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super(PfomMMVAE, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = PfomMMDiv()

    def encode(self, input_batch: typing.Mapping[str, Tensor]) -> typing.Mapping[str, EncModPFoM]:
        """
        Encoder that outputs parameters for base distribution of z and flow parameters.
        """
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                x_m = input_batch[mod_str]
                style_mu, style_logvar, class_mu, class_logvar, h = mod.encoder(x_m)
                latents_class = Distr(mu=class_mu, logvar=class_logvar)

                enc_mods[mod_str] = EncModPFoM(latents_class=latents_class, h=h)
                if style_mu:
                    latents_style = Distr(mu=style_mu, logvar=style_logvar)

                    enc_mods[mod_str].latents_style = latents_style

        return enc_mods

    def mixture_component_selection(self, enc_mods: Mapping[str, EncModPFoM], s_key: str) -> typing.Tuple[
        Distr, PlanarFlowParams]:
        """For each element in batch select an expert from subset with equal probability."""
        num_samples = enc_mods[list(enc_mods)[0]].latents_class.mu.shape[0]
        mods = self.subsets[s_key]

        # fuse the subset
        joint_mus = torch.Tensor().to(self.flags.device)
        joint_logvars = torch.Tensor().to(self.flags.device)
        joint_h = torch.Tensor().to(self.flags.device)

        if self.flags.weighted_mixture:
            # define confidence of expert by the inverse of the logvar.
            # Sample experts with probability proportional to confidence.
            confidences = [1 / enc_mods[mod.name].latents_class.logvar.mean().abs() for mod in mods]
            bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
            if sum(bins) != num_samples:
                bins[confidences.index(max(confidences))] += num_samples - sum(bins)
        else:
            # fill zk_subset with an equal amount of each expert
            bins = split_int_to_bins(number=num_samples, nbr_bins=len(mods))

        # get latents and flow params for each expert
        for chunk_size, mod in zip(bins, mods):
            expert = enc_mods[mod.name]
            joint_mus = torch.cat(
                (joint_mus, expert.latents_class.mu[joint_mus.shape[0]:joint_mus.shape[0] + chunk_size]),
                dim=0)
            joint_logvars = torch.cat(
                (joint_logvars,
                 expert.latents_class.logvar[
                 joint_logvars.shape[0]:joint_logvars.shape[0] + chunk_size]),
                dim=0)
            joint_h = torch.cat((joint_h, expert.h[joint_h.shape[0]:joint_h.shape[0] + chunk_size]))

        if self.num_flows:
            # get amortized parameters
            joint_flow_params = PlanarFlowParams(**{
                'u': self.amor_u(joint_h).view(joint_h.shape[0], self.num_flows, self.flags.class_dim, 1),
                'w': self.amor_w(joint_h).view(joint_h.shape[0], self.num_flows, 1, self.flags.class_dim),
                'b': self.amor_b(joint_h).view(joint_h.shape[0], self.num_flows, 1, 1)})
        else:
            joint_flow_params = {k: None for k in ['u', 'w', 'b']}

        assert joint_mus.shape == joint_logvars.shape == torch.Size([num_samples, self.flags.class_dim])

        # normalize latents by number of modalities in subset
        # todo: this is not necessary?
        # weights_subset = ((1 / float(len(mods))) * torch.ones_like(joint_logvars).to(self.flags.device))

        return Distr(mu=joint_mus, logvar=joint_logvars), joint_flow_params

    def fuse_modalities(self, enc_mods: Mapping[str, EncModPFoM],
                        batch_mods: typing.Iterable[str]) -> JointLatentsPFoM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        distr_subsets = {}

        # concatenate mus and logvars for every modality in each subset
        for s_key in batch_subsets:
            subset_distr, subset_flow_params = self.mixture_component_selection(enc_mods, s_key)

            z0, zk, log_det_j = self.apply_flow(subset_distr, subset_flow_params)

            distr_subsets[s_key] = SubsetPFoM(z0, zk, log_det_j)

            if len(self.subsets[s_key]) == len(batch_mods):
                joint_embedding = JointEmbeddingPFoM(embedding=zk, mod_strs=s_key.split('_'), log_det_j=log_det_j)

        return JointLatentsPFoM(joint_embedding=joint_embedding, subsets=distr_subsets)


class PlanarMixtureMMVae(PlanarFlowMMVAE):
    """
    Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
    Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
    """

    def __init__(self, exp, flags, modalities, subsets):
        super(PlanarMixtureMMVae, self).__init__(exp, flags, modalities, subsets)
        self.mm_div = PlanarMixtureMMDiv()

    def encode(self, input_batch: Mapping[str, Tensor]) -> Mapping[str, EncModPlanarMixture]:
        """
        Encoder that outputs parameters for base distribution of z and flow parameters.
        """
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                x_m = input_batch[mod_str]
                style_mu, style_logvar, class_mu, class_logvar, h = mod.encoder(x_m)
                latents_class = Distr(mu=class_mu, logvar=class_logvar)

                # get amortized u an w for all flows
                if self.num_flows:
                    flow_params = {
                        'u': self.amor_u(h).view(h.shape[0], self.num_flows, self.flags.class_dim, 1),
                        'w': self.amor_w(h).view(h.shape[0], self.num_flows, 1, self.flags.class_dim),
                        'b': self.amor_b(h).view(h.shape[0], self.num_flows, 1, 1)}
                else:
                    flow_params = {k: None for k in ['u', 'w', 'b']}

                enc_mods[mod_str] = EncModPlanarMixture(latents_class=latents_class,
                                                        flow_params=PlanarFlowParams(**flow_params))
                if style_mu:
                    latents_style = Distr(mu=style_mu, logvar=style_logvar)

                    enc_mods[mod_str].latents_style = latents_style

        # pass the latents of each class through the flow
        enc_mods = self.apply_flows(enc_mods)

        return enc_mods

    def apply_flows(self, enc_mods: Mapping[str, EncModPlanarMixture]) -> Mapping[str, EncModPlanarMixture]:
        """Apply the flow for each modality."""
        for mod_str, enc_mod in enc_mods.items():
            latents_class = enc_mod.latents_class
            log_det_j = torch.zeros(latents_class.mu.shape[0]).to(self.flags.device)
            flow_params = enc_mod.flow_params
            # Sample z_0
            z = [latents_class.reparameterize()]

            # Normalizing flows
            for k in range(self.num_flows):
                flow_k = getattr(self, 'flow_' + str(k))
                # z' = z + u h( w^T z + b)
                z_k, log_det_jacobian = flow_k(z[k], flow_params.u[:, k, :, :], flow_params.w[:, k, :, :],
                                               flow_params.b[:, k, :, :])
                z.append(z_k)
                log_det_j += log_det_jacobian
            enc_mods[mod_str].z0 = z[0]
            enc_mods[mod_str].zk = z[-1]
            enc_mods[mod_str].log_det_j = log_det_j
        return enc_mods

    def fuse_modalities(self, enc_mods: Mapping[str, EncModPlanarMixture],
                        batch_mods: typing.Iterable[str]) -> JointLatentsPlanarMixture:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        distr_subsets = {}

        # concatenate mus and logvars for every modality in each subset
        for s_key in batch_subsets:
            z_subset = self.mixture_component_selection(enc_mods, s_key)
            distr_subsets[s_key] = z_subset

            if len(self.subsets[s_key]) == len(batch_mods):
                joint_embedding = JointEmbeddingPlanarMixture(embedding=z_subset, mod_strs=s_key.split('_'))

        return JointLatentsPlanarMixture(joint_embedding=joint_embedding, subsets=distr_subsets)

    def fuse_subset(self, enc_mods: Mapping[str, EncModPlanarMixture], s_key: str) -> Distr:
        """Fuse encoded modalities in subset."""
        mods = self.subsets[s_key]
        # fuse the subset
        zk_subset = torch.Tensor().to(self.flags.device)

        # get enc_mods for subset
        for mod in mods:
            zk_subset = torch.cat((zk_subset, enc_mods[mod.name].zk.unsqueeze(0)), dim=0)

        # normalize latents by number of modalities in subset
        weights_subset = ((1 / float(len(mods))) * torch.ones_like(zk_subset).to(self.flags.device))

        # fuse enc_mods
        return (weights_subset * zk_subset).sum(dim=0)

    def mixture_component_selection(self, enc_mods: Mapping[str, EncModPlanarMixture], s_key: str,
                                    weight_joint: bool = True) -> Distr:
        """For each element in batch select an expert from subset with equal probability."""
        num_samples = enc_mods[list(enc_mods)[0]].zk.shape[0]
        mods = self.subsets[s_key]
        # fuse the subset
        zk_subset = torch.Tensor().to(self.flags.device)

        if self.flags.weighted_mixture:
            # define confidence of expert by the mean of zk. Sample experts with probability proportional to confidence.
            confidences = [enc_mods[mod.name].zk.mean().abs() for mod in mods]
            bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
            if sum(bins) != num_samples:
                bins[confidences.index(max(confidences))] += num_samples - sum(bins)
        else:
            # fill zk_subset with an equal amount of each expert
            bins = split_int_to_bins(number=num_samples, nbr_bins=len(mods))

        # get enc_mods for subset
        for chunk_size, mod in zip(bins, mods):
            zk_subset = torch.cat(
                (zk_subset, enc_mods[mod.name].zk[zk_subset.shape[0]:zk_subset.shape[0] + chunk_size]), dim=0)

        assert zk_subset.shape == torch.Size([num_samples, self.flags.class_dim])

        if weight_joint:
            # normalize latents by number of modalities in subset
            weights_subset = ((1 / float(len(mods))) * torch.ones_like(zk_subset).to(self.flags.device))

            return (weights_subset * zk_subset)
        else:
            return zk_subset

    # def mixture_component_selection(self, enc_mods: Mapping[str, EncModPlanarMixture], s_key: str) -> Distr:
    #     num_samples = enc_mods[list(enc_mods)[0]].zk.shape[0]
    #     mods = self.subsets[s_key]
    #
    #     w_modalities = torch.Tensor(self.flags.alpha_modalities).to(self.flags.device)
    #
    #     idx_start = []
    #     idx_end = []
    #     for k in range(len(mods)):
    #         i_start = 0 if k == 0 else int(idx_end[k - 1])
    #         if k == w_modalities.shape[0] - 1:
    #             i_end = num_samples
    #         else:
    #             i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
    #         idx_start.append(i_start)
    #         idx_end.append(i_end)
    #
    #     idx_end[-1] = num_samples
    #
    #     zk_sel = torch.cat([enc_mods[mod.name].zk[idx_start[k]:idx_end[k], :] for k, mod in enumerate(mods)])
    #
    #     return zk_sel
