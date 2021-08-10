import typing

from mmvae_hub.evaluation.divergence_measures.mm_div import GfMMMDiv, GfMoPDiv, PGfMMMDiv, BaseMMDiv, MoFoGfMMMDiv, \
    BMoGfMMMDiv, GfMMMDiv_old
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.MixtureVaes import JointElboMMVae
from mmvae_hub.networks.flows.AffineFlows import AffineFlow
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods, mixture_component_selection_embedding


class GfMVAE(BaseMMVAE):
    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = GfMMMDiv()

        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

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


class MoGfMVAE_old(BaseMMVAE):
    "Mixture of Generalized f-Means VAE"

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = GfMMMDiv_old()

        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

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

        z_joint = mixture_component_selection_embedding(subset_embeds=subset_embeddings, s_key='all', flags=self.flags)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsMoGfM(joint_embedding=joint_embedding, subsets=subset_embeddings, subset_samples=None)


class iwMoGfMVAE(BaseMMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.num_samples = 100
        self.mm_div = GfMMMDiv(flags=flags, num_samples=self.num_samples)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        transformed_enc_mods = {}
        subset_embeddings = {}
        subset_samples = {}

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.mu.shape[0]

        transformed_enc_mods = {
            mod_key: self.flow(
                torch.cat(tuple(distr.latents_class.reparameterize().unsqueeze(dim=0) for _ in range(self.num_samples)),
                          dim=0).reshape((self.num_samples * batch_size, self.flags.class_dim)))[
                0] for mod_key, distr in
            enc_mods.items()}

        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod.name].unsqueeze(dim=0)), dim=0)
            # mean of zks
            z_mean = torch.mean(subset_zks, dim=0)
            # calculate inverse flow
            samples = self.flow.rev(z_mean)[0]

            samples = samples
            subset_samples[s_key] = samples

            # subset_embeddings[s_key] = samples.mean(dim=0)

        selection = mixture_component_selection_embedding(subset_embeds=subset_samples, s_key='all', flags=self.flags).reshape((self.num_samples, batch_size, self.flags.class_dim))
        z_joint = selection.mean(dim=0)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])
        subset_samples = {k:samples.reshape((self.num_samples, batch_size, self.flags.class_dim)) for k, samples in subset_samples.items()}

        return JointLatentsMoGfM(joint_embedding=joint_embedding, subsets={k:samples.mean(dim=0) for k, samples in subset_samples.items()},
                                 subset_samples=subset_samples, enc_mods=enc_mods)


class MoGfMVAE(BaseMMVAE):
    "Mixture of Generalized f-Means VAE"

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.num_samples = 100
        self.mm_div = GfMMMDiv(flags=flags, num_samples=self.num_samples)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        transformed_enc_mods = {}
        subset_embeddings = {}
        subset_samples = {}

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.mu.shape[0]

        # sample num_samples from enc_mods and pass them through flow
        # for mod_key, distr in enc_mods.items():
        #     transformed_samples = torch.Tensor().to(self.flags.device)
        #     enc_mods_samples = torch.cat(tuple(distr.latents_class.reparameterize().unsqueeze(dim=0)
        #                                        for _ in range(self.num_samples)), dim=0) \
        #         .reshape((self.num_samples * batch_size, self.flags.class_dim))
        # pass samples batchwise through flow to limit gpu mem usage
        # for batch in chunks(enc_mods_samples, 100):
        #     transformed_samples = torch.cat((transformed_samples, self.flow(batch)[0]))
        # transformed_enc_mods[mod_key] = transformed_samples

        transformed_enc_mods = {
            mod_key: self.flow(
                torch.cat(tuple(distr.latents_class.reparameterize().unsqueeze(dim=0) for _ in range(self.num_samples)),
                          dim=0).reshape((self.num_samples * batch_size, self.flags.class_dim)))[
                0] for mod_key, distr in
            enc_mods.items()}

        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod.name].unsqueeze(dim=0)), dim=0)
            # mean of zks
            z_mean = torch.mean(subset_zks, dim=0)
            # calculate inverse flow
            samples = self.flow.rev(z_mean)[0]

            # samples = torch.Tensor().to(self.flags.device)
            # for batch in chunks(z_mean, 100):
            #     samples = torch.cat((samples, self.flow.rev(batch)[0]))

            samples = samples.reshape((self.num_samples, batch_size, self.flags.class_dim))
            subset_samples[s_key] = samples

            subset_embeddings[s_key] = samples.mean(dim=0)

        z_joint = mixture_component_selection_embedding(subset_embeds=subset_embeddings, s_key='all', flags=self.flags)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsMoGfM(joint_embedding=joint_embedding, subsets=subset_embeddings,
                                 subset_samples=subset_samples, enc_mods=enc_mods)


class MoFoGfMVAE(BaseMMVAE):
    """Mixture of Flow of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = MoFoGfMMMDiv()

        self.gfm_flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)
        self.flow = AffineFlow(flags.class_dim, flags.num_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsMoFoGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        subset_embeddings = {}

        # pass all experts through the flow
        enc_mod_zks = {mod_key: self.gfm_flow(distr.latents_class.reparameterize())[0] for mod_key, distr in
                       enc_mods.items()}

        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, enc_mod_zks[mod.name].unsqueeze(dim=0)), dim=0)
            # mean of zks
            z_mean = torch.mean(subset_zks, dim=0)
            # calculate inverse flow
            z0_subset, _ = self.gfm_flow.rev(z_mean)

            # pass subset through flow
            zk_subset, log_det_j = self.flow.forward(z0_subset)

            subset_embeddings[s_key] = SubsetMoFoGfM(z0=z0_subset, zk=zk_subset, log_det_j=log_det_j)

        # select expert for z_joint
        subsets = {k: v.zk for k, v in subset_embeddings.items()}
        z_joint = mixture_component_selection_embedding(subset_embeds=subsets, s_key='all', flags=self.flags)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsMoFoGfM(joint_embedding=joint_embedding, subsets=subset_embeddings)


class BMoGfMVAE(MoGfMVAE):
    """ Bounded Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)
        self.mm_div = BMoGfMMMDiv()


class GfMoPVAE(JointElboMMVae):
    """GfM of Product of experts VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = GfMoPDiv()

        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

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
        self.flow_mus = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)
        self.flow_logvars = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatents:
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


class MopGfM(PGfMVAE):
    """Mixture of parameter GfM method."""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)
        self.mm_div = BaseMMDiv()

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatents:
        # apply pgfm method
        subset_distrs = super().fuse_modalities(enc_mods, batch_mods).subsets

        # select expert for z_joint
        # joint_distr = mixture_component_selection(distrs=subset_distrs, s_key='all', flags=self.flags)
        mus = torch.cat(tuple(distr.mu.unsqueeze(0) for _, distr in subset_distrs.items()), dim=0)
        logvars = torch.cat(tuple(distr.logvar.unsqueeze(0) for _, distr in subset_distrs.items()), dim=0)

        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(self.flags.device)
        joint_distr = self.moe_fusion(mus, logvars, weights)

        return JointLatents(batch_mods, joint_distr=joint_distr, subsets=subset_distrs)


class PGfMoPVAE(BaseMMVAE):
    """
    Params Generalized f-Means of Product of Experts VAE: class of methods where the means and logvars of all experts
    are fused with a generalized f-mean.
    """

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        self.mm_div = PGfMMMDiv()
        self.flow_mus = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)
        self.flow_logvars = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

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
