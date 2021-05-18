# -*- coding: utf-8 -*-
import typing
from abc import abstractmethod, ABC
from typing import Mapping

import torch
from torch import nn as nn, Tensor

from mmvae_hub.base.evaluation.divergence_measures.mm_div import FlowMMDiv
from mmvae_hub.base.utils.Dataclasses import EncModPlanarMixture, Distr, PlanarFlowParams, JointLatentsPlanarMixture, \
    JointEmbeddingPlanarMixture
from mmvae_hub.base.utils.fusion_functions import subsets_from_batchmods
from mmvae_hub.base.utils.utils import split_int_to_bins
from mmvae_hub.networks.MixtureVaes import MOEMMVae
from mmvae_hub.networks.utils import flows


class BaseFlowMMVAE(MOEMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        super(BaseFlowMMVAE, self).__init__(exp, flags, modalities, subsets)
        self.flow = None

    @abstractmethod
    def encode(self, input_batch) -> dict:
        pass


class PlanarMixtureMMVae(BaseFlowMMVAE, ABC):
    """
    Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
    Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
    """

    def __init__(self, exp, flags, modalities, subsets):
        super(PlanarMixtureMMVae, self).__init__(exp, flags, modalities, subsets)

        self.mm_div = FlowMMDiv()

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
        A joint latent space is then created by fusing all subspaces.
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

    def mixture_component_selection(self, enc_mods: Mapping[str, EncModPlanarMixture], s_key: str) -> Distr:
        """For each element in batch select an expert from subset with equal probability."""
        num_samples = enc_mods[list(enc_mods)[0]].zk.shape[0]
        mods = self.subsets[s_key]
        # fuse the subset
        zk_subset = torch.Tensor().to(self.flags.device)

        if self.flags.weighted_mixture:
            # define confidence of expert by the mean of zk. Sample experts with probability proportional to confidence.
            confidences = [enc_mods[mod.name].zk.mean().abs() for mod in mods]
            bins = [int(num_samples * (conf / sum(confidences))) for conf in confidences]
            if sum(bins) != len(mods):
                bins[confidences.index(max(confidences))] += 1
        else:
            # fill zk_subset with an equal amount of each expert
            bins = split_int_to_bins(number=num_samples, nbr_bins=len(mods))

        # get enc_mods for subset
        for chunk_size, mod in zip(bins, mods):
            zk_subset = torch.cat(
                (zk_subset, enc_mods[mod.name].zk[zk_subset.shape[0]:zk_subset.shape[0] + chunk_size]), dim=0)

        assert zk_subset.shape == torch.Size([num_samples, self.flags.class_dim])

        # normalize latents by number of modalities in subset
        weights_subset = ((1 / float(len(mods))) * torch.ones_like(zk_subset).to(self.flags.device))

        return (weights_subset * zk_subset)