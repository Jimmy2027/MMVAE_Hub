import typing
from typing import Mapping

import torch
from torch import Tensor

from mmvae_hub.evaluation.divergence_measures.mm_div import GfMMMDiv, GfMoPDiv, PGfMMMDiv
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.MixtureVaes import JointElboMMVae
from mmvae_hub.networks.flows.AffineFlows import AffineFlow
from mmvae_hub.utils.Dataclasses import BaseEncMod, JointLatentsGfM, JointEmbeddingFoEM, Distr, EncModGfM, \
    JointLatentsGfMoP, JointLatentsFoEM, JointLatents
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods, mixture_component_selection_embedding


class GfMVAE(BaseMMVAE):
    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = GfMMMDiv()

        self.flow = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        subset_embeddings = {}

        # pass all experts through the flow
        enc_mod_zks = {mod_key: self.flow(distr.latents_class.reparameterize())[0] for mod_key, distr in
                       enc_mods.items()}

        # concatenate mus and logvars for every modality in each subset
        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, enc_mod_zks[mod.name].unsqueeze(dim=0)), dim=0)
            # mean of zks
            z_mean = torch.mean(subset_zks, dim=0)
            # calculate inverse flow
            z_subset, _ = self.flow.rev(z_mean)

            subset_embeddings[s_key] = z_subset

            if len(self.subsets[s_key]) == len(batch_mods):
                joint_embedding = JointEmbeddingFoEM(embedding=z_subset, mod_strs=s_key.split('_'))

        return JointLatentsGfM(joint_embedding=joint_embedding, subsets=subset_embeddings)


class GfMoPVAE(JointElboMMVae):
    """GfM of Product of experts VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = GfMoPDiv()

        self.flow = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        distr_subsets = {}

        for s_key in batch_subsets:
            distr_subset = self.fuse_subset(enc_mods, s_key)
            distr_subsets[s_key] = distr_subset

        z_joint = self.gfm(distr_subsets)

        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs='joint')

        return JointLatentsGfMoP(joint_embedding=joint_embedding, subsets=distr_subsets)

    def gfm(self, distr_subsets: Mapping[str, Distr]) -> Tensor:
        """Merge subsets with a generalized f mean."""
        # pass all subset experts through the flow
        enc_subset_zks = {s_key: self.encode_expert(distr_subset) for s_key, distr_subset in distr_subsets.items()}

        # get the mixture of each encoded subset
        z_mixture = mixture_component_selection_embedding(enc_mods=enc_subset_zks, s_key='all', flags=self.flags)
        # question: should I normalize by the number of modalities here?

        # pass the mixture backwards through the flow.
        z_joint, _ = self.flow.rev(z_mixture)
        return z_joint

    def encode_expert(self, expert_distr: Distr) -> EncModGfM:
        zk, _ = self.flow(expert_distr.reparameterize())
        return EncModGfM(zk=zk)


class PGfMVAE(BaseMMVAE):
    """
    Params Generalized f-Means VAE: class of methods where the means and logvars of all experts are fused with a
    generalized f-mean.
    """

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = PGfMMMDiv()
        self.flow_mus = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)
        self.flow_logvars = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsFoEM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        distr_subsets = {}

        # pass all experts through the flow
        enc_mod_transformed = {mod_key: self.encode_expert(enc_mod.latents_class) for mod_key, enc_mod in
                               enc_mods.items()}

        # concatenate mus and logvars for every modality in each subset
        for s_key in batch_subsets:
            if len(self.subsets[s_key]) == 1:
                q_subset = enc_mods[s_key].latents_class
            else:
                q_subset = self.gfm(distrs=[enc_mod_transformed[mod.name] for mod in self.subsets[s_key]])
            distr_subsets[s_key] = q_subset

            if len(self.subsets[s_key]) == len(batch_mods):
                joint_distr = q_subset
                fusion_subsets_keys = s_key.split('_')
                joint_distr.mod_strs = fusion_subsets_keys

        return JointLatents(fusion_subsets_keys, joint_distr=joint_distr, subsets=distr_subsets)

    def gfm(self, distrs: typing.List[Distr]) -> Distr:
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)

        for distr in distrs:
            mus = torch.cat((mus, distr.mu.unsqueeze(dim=0)), dim=0)
            logvars = torch.cat((logvars, distr.logvar.unsqueeze(dim=0)), dim=0)

        mu_average = torch.mean(mus, dim=0)
        logvar_average = torch.mean(logvars, dim=0)

        mu_gfm, _ = self.flow_mus.rev(mu_average)
        logvar_gfm, _ = self.flow_logvars.rev(logvar_average)
        return Distr(mu=mu_gfm, logvar=logvar_gfm)

    def encode_expert(self, expert_distr: Distr) -> Distr:
        mu_k, _ = self.flow_mus(expert_distr.mu)
        logvar_k, _ = self.flow_logvars(expert_distr.logvar)
        return Distr(mu=mu_k, logvar=logvar_k)


class PGfMoPVAE(BaseMMVAE):
    """
    Params Generalized f-Means of Product of Experts VAE: class of methods where the means and logvars of all experts are fused with a
    generalized f-mean.
    """

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = PGfMMMDiv()
        self.flow_mus = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)
        self.flow_logvars = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatents:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        distr_subsets = {}
        fusion_subsets_keys = []

        for s_key in batch_subsets:
            distr_subset = self.fuse_subset(enc_mods, s_key)
            distr_subsets[s_key] = distr_subset
            fusion_subsets_keys.append(s_key)

        joint_distr = self.gfm(distr_subsets)

        return JointLatents(fusion_subsets_keys, joint_distr=joint_distr, subsets=distr_subsets)

    def gfm(self, distrs: Mapping[str, Distr]) -> Distr:
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)

        # pass all mus and sigmas of subsets through flows and concatenate them
        for _, distr in distrs.items():
            mu_k, _ = self.flow_mus(distr.mu)
            logvar_k, _ = self.flow_logvars(distr.logvar)
            mus = torch.cat((mus, mu_k.unsqueeze(dim=0)), dim=0)
            logvars = torch.cat((logvars, logvar_k.unsqueeze(dim=0)), dim=0)

        mu_average = torch.mean(mus, dim=0)
        logvar_average = torch.mean(logvars, dim=0)

        mu_gfm, _ = self.flow_mus.rev(mu_average)
        logvar_gfm, _ = self.flow_logvars.rev(logvar_average)
        return Distr(mu=mu_gfm, logvar=logvar_gfm)
