# -*- coding: utf-8 -*-
import typing
from abc import abstractmethod

import torch

from mmvae_hub.evaluation.divergence_measures.mm_div import PlanarMixtureMMDiv, PfomMMDiv, PoPEMMDiv
from mmvae_hub.networks.Flows import PlanarFlow
from mmvae_hub.networks.MixtureVaes import MOEMMVae
from mmvae_hub.networks.PoEMMVAE import POEMMVae
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods
from mmvae_hub.utils.utils import split_int_to_bins


class FlowVAE:
    """Class of methods where a flow is applied on the latent distributions."""

    def __init__(self):
        pass


class JointFromFlowVAE(FlowVAE):
    """Class of methods where the flow is applied on each modality."""

    def __init__(self):
        super().__init__()


class FlowOfJointVAE(FlowVAE):
    """Class of methods where the flow is applied on a joint distribution."""

    def __init__(self):
        super().__init__()

    def fuse_modalities(self, enc_mods: Mapping[str, EncModPFoM],
                        batch_mods: typing.Iterable[str]) -> JointLatentsPFoM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        distr_subsets = {}

        # concatenate mus and logvars for every modality in each subset
        for s_key in batch_subsets:
            subset_distr, subset_flow_params = self.expert_fusion(enc_mods, s_key)

            z0, zk, log_det_j = self.flow.forward(subset_distr, subset_flow_params)

            distr_subsets[s_key] = SubsetPFoM(q0=subset_distr, z0=z0, zk=zk, log_det_j=log_det_j)

            if len(self.subsets[s_key]) == len(batch_mods):
                joint_embedding = JointEmbeddingPFoM(embedding=zk, mod_strs=s_key.split('_'), log_det_j=log_det_j)

        return JointLatentsPFoM(joint_embedding=joint_embedding, subsets=distr_subsets)

    @abstractmethod
    def expert_fusion(self, enc_mods, s_key):
        pass


class PoPE(FlowOfJointVAE, POEMMVae):
    """Planar flow Of Product of Experts."""

    def __init__(self, exp, flags, modalities, subsets):
        FlowOfJointVAE.__init__(self)
        POEMMVae.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = PoPEMMDiv()
        self.flow = PlanarFlow(flags)

    def expert_fusion(self, enc_mods, s_key):
        # armotized flow params are not implemented for PoPE
        subset_flow_params = None
        return self.fuse_subset(enc_mods, s_key), subset_flow_params


class PfomMMVAE(FlowOfJointVAE, MOEMMVae):
    """Planar Flow of Mixture multi-modal VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        FlowOfJointVAE.__init__(self)
        MOEMMVae.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = PfomMMDiv()
        self.flow = PlanarFlow(flags)

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

        joint_flow_params = self.flow.get_flow_params(joint_h)

        assert joint_mus.shape == joint_logvars.shape == torch.Size([num_samples, self.flags.class_dim])

        # normalize latents by number of modalities in subset
        # todo: this is not necessary?
        # weights_subset = ((1 / float(len(mods))) * torch.ones_like(joint_logvars).to(self.flags.device))

        return Distr(mu=joint_mus, logvar=joint_logvars), joint_flow_params

    def expert_fusion(self, enc_mods, s_key):
        return self.mixture_component_selection(enc_mods, s_key)


class PlanarMixtureMMVae(JointFromFlowVAE, MOEMMVae):
    """
    Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
    Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
    """

    def __init__(self, exp, flags, modalities, subsets):
        JointFromFlowVAE.__init__(self)
        MOEMMVae.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = PlanarMixtureMMDiv()
        self.flow = PlanarFlow(flags)

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

                flow_params = self.flow.get_flow_params(h)

                enc_mods[mod_str] = EncModPlanarMixture(latents_class=latents_class,
                                                        flow_params=flow_params)
                if style_mu:
                    latents_style = Distr(mu=style_mu, logvar=style_logvar)

                    enc_mods[mod_str].latents_style = latents_style

        # pass the latents of each class through the flow
        enc_mods = self.apply_flows(enc_mods)

        return enc_mods

    def apply_flows(self, enc_mods: Mapping[str, EncModPlanarMixture]) -> Mapping[str, EncModPlanarMixture]:
        """Apply the flow for each modality."""
        for mod_str, enc_mod in enc_mods.items():
            enc_mods[mod_str].z0, enc_mods[mod_str].zk, enc_mods[mod_str].log_det_j = self.flow.forward(
                enc_mod.latents_class, enc_mod.flow_params)

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
