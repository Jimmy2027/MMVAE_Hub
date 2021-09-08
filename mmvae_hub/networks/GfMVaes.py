import typing

import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from mmvae_hub.evaluation.divergence_measures.mm_div import GfMMMDiv, GfMoPDiv, PGfMMMDiv, BaseMMDiv, MoFoGfMMMDiv, \
    BMoGfMMMDiv, GfMMMDiv_old
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.MixtureVaes import MoPoEMMVae
from mmvae_hub.networks.flows.AffineFlows import AffineFlow
from mmvae_hub.networks.iwVaes import log_mean_exp, iwMMVAE
from mmvae_hub.utils.dataclasses.iwdataclasses import *
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


class iwMoGfMVAE(iwMMVAE, BaseMMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)
        self.mm_div = GfMMMDiv(flags=flags, K=self.K)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

    def reparam_with_eps(self, distr: Distr, eps: Tensor):
        """Apply the reparameterization trick on distr with given epsilon"""
        std = distr.logvar.mul(0.5).exp_()
        # print(std.max())
        return eps.mul(std).add_(distr.mu)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        subset_samples = {}

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.mu.shape[0]

        # sample K*bs samples from prior
        epss = MultivariateNormal(torch.zeros(self.flags.class_dim, device=self.flags.device),
                                  torch.eye(self.flags.class_dim, device=self.flags.device)). \
            sample((self.K * batch_size,)).reshape((self.K, batch_size, self.flags.class_dim))

        # temp
        transformed_enc_mods = {
            mod_key: self.flow(
                torch.cat(tuple(self.reparam_with_eps(distr.latents_class, epss[k_idx]).unsqueeze(dim=0) for k_idx in
                                range(self.K)),
                          dim=0).reshape((self.K * batch_size, self.flags.class_dim)))[
                0] for mod_key, distr in
            enc_mods.items()}

        # transformed_enc_mods = {
        #     mod_key: self.flow(
        #         torch.cat(tuple(distr.latents_class.reparameterize().unsqueeze(dim=0) for k_idx in
        #                         range(self.K)),
        #                   dim=0).reshape((self.K * batch_size, self.flags.class_dim)))[
        #         0] for mod_key, distr in
        #     enc_mods.items()}

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

        subset_samples = {k: samples.reshape((self.K, batch_size, self.flags.class_dim)) for k, samples in
                          subset_samples.items()}

        z_joint = torch.cat([v.squeeze() for _, v in subset_samples.items()], dim=0)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM(joint_embedding=joint_embedding,
                                   subsets=subset_samples,
                                   subset_samples=subset_samples, enc_mods=enc_mods, epss=epss)

    def decode(self, enc_mods: Mapping[str, BaseEncMod], joint_latents: iwJointLatents) -> dict:
        """Decoder outputs each reconstructed modality as a dict."""
        rec_mods = {}
        for subset_str, subset in joint_latents.subsets.items():
            rec_mods[subset_str] = {}
            for out_mod_str, dec_mod in self.modalities.items():
                mu, logvar = dec_mod.decoder(None,
                                             subset.reshape(
                                                 (self.K * self.flags.batch_size, self.flags.class_dim)))
                rec_mods[subset_str][out_mod_str] = distr.Laplace(
                    loc=mu.reshape((self.K, self.flags.batch_size, *mu.shape[1:])), scale=logvar)

        return rec_mods

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        epss = forward_results.joint_latents.epss
        losses = []
        klds = {}
        log_probs = {}
        for mod_str, subset_samples in subsets.items():
            # kl_div = subset_sample * (log(subset_sample) - log(eps))
            # subset_samples = subset_samples.flatten(start_dim=0, end_dim=1)
            # kl_div = torch.where(subset_samples.round().type(torch.double) != float(0), (
            #             subset_samples * torch.log(subset_samples / epss.flatten(start_dim=0, end_dim=1))).type(
            #     torch.double), float(0)).sum(-1).reshape((self.flags.K, self.flags.batch_size))

            # temp
            # print((subset_samples == 0).sum())
            # kl_div = torch.nn.functional.kl_div(subset_samples, subset_samples, reduce=False, log_target=True).mean(-1)
            # kl_div = torch.nn.functional.kl_div(epss, subset_samples, reduce=False, log_target=True).mean(-1)
            epss = torch.where(epss.abs() <= 0.001, torch.tensor(0.01, device=self.flags.device), epss)
            # print(subset_samples.max(), subset_samples.min(), subset_samples.abs().min())
            interm1 = (subset_samples / epss).abs() + 1e-4
            print('interm1: ', interm1.min(), interm1.max())
            interm2 = torch.log(interm1)
            print('interm2: ', interm2.min(), interm2.max())
            kl_div = (subset_samples * interm2).mean(-1)
            print('kl_div: ', kl_div.min(), kl_div.max())
            # kl_div = torch.nn.functional.kl_div(subset_samples, subset_samples, reduce=False, log_target=True).mean(-1)
            # kl_div = torch.nn.functional.kl_div(epss, subset_samples, reduction='batchmean', log_target=True).mean(-1)
            # print('kl_div', kl_div.max(),kl_div.min() )
            # temp
            # assert kl_div.max() <= 10e5
            # kl_div = torch.where(kl_div >= 10e5, torch.tensor(10.0, device=self.flags.device), kl_div)

            # kl_div = torch.nn.functional.kl_div(epss, epss, reduce=False, log_target=True).sum(-1)
            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[mod_str].items()]

            lpx_z = torch.stack(lpx_z).sum(0)
            # temp
            loss = lpx_z + self.flags.beta * kl_div
            # loss = lpx_z + 0 * kl_div
            # print(loss)
            # loss = lpx_z + (self.flags.beta * kl_div) if self.flags.beta else lpx_z
            losses.append(loss)
            log_probs[mod_str] = lpx_z.mean()
            try:
                klds[mod_str] = self.flags.beta * log_mean_exp(kl_div).sum()
            except:
                klds[mod_str] = kl_div

        # print('losses', losses)
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()
        # print(total_loss)
        # print(total_loss)
        # joint_div average of all subset divs
        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))
        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds

    def conditioned_generation(self, input_samples: dict, subset_key: str, style=None):
        """
        Generate samples conditioned with input samples for a given subset.

        subset_key str: The key indicating which subset is used for the generation.
        """

        # infer latents from batch
        enc_mods, joint_latents = self.inference(input_samples)

        subset_embedding = joint_latents.subsets[subset_key].mean(dim=0)
        cond_mod_in = ReparamLatent(content=subset_embedding, style=style)
        return self.generate_from_latents(cond_mod_in)


class iwMoGfMVAE_(iwMMVAE, BaseMMVAE):
    """Try iwMoGfMVAE without flows for debugging."""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)
        self.mm_div = GfMMMDiv(flags=flags, K=self.K)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

    def reparam_with_eps(self, distr: Distr, eps: Tensor):
        """Apply the reparameterization trick on distr with given epsilon"""
        std = distr.logvar.mul(0.5).exp_()
        return eps.mul(std).add_(distr.mu)

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

        # sample K*bs samples from prior
        epss = MultivariateNormal(torch.zeros(self.flags.class_dim, device=self.flags.device),
                                  torch.eye(self.flags.class_dim, device=self.flags.device)). \
            sample((self.K * batch_size,)).reshape((self.K, batch_size, self.flags.class_dim))

        transformed_enc_mods = {
            mod_key:
                torch.cat(tuple(distr.latents_class.reparameterize().unsqueeze(dim=0) for k_idx in
                                range(self.K)),
                          dim=0).reshape((self.K * batch_size, self.flags.class_dim)) for mod_key, distr in
            enc_mods.items()}

        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod.name].unsqueeze(dim=0)), dim=0)
            # mean of zks
            # z_mean = torch.mean(subset_zks, dim=0)
            z_mean = subset_zks[0]
            # z_mean = torch.mean(subset_zks, dim=0)
            # calculate inverse flow
            samples = z_mean

            assert not samples.isnan().all()

            samples = samples
            subset_samples[s_key] = samples

            # subset_embeddings[s_key] = samples.mean(dim=0)

        subset_samples = {k: samples.reshape((self.K, batch_size, self.flags.class_dim)) for k, samples in
                          subset_samples.items()}

        z_joint = torch.cat([v.squeeze() for _, v in subset_samples.items()], dim=0)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM(joint_embedding=joint_embedding,
                                   subsets=subset_samples,
                                   subset_samples=subset_samples, enc_mods=enc_mods, epss=epss)

    def decode(self, enc_mods: Mapping[str, BaseEncMod], joint_latents: iwJointLatents) -> dict:
        """Decoder outputs each reconstructed modality as a dict."""
        rec_mods = {}
        for subset_str, subset in joint_latents.subsets.items():
            rec_mods[subset_str] = {}
            for out_mod_str, dec_mod in self.modalities.items():
                mu, logvar = dec_mod.decoder(None,
                                             subset.reshape(
                                                 (self.K * self.flags.batch_size, self.flags.class_dim)))
                rec_mods[subset_str][out_mod_str] = distr.Laplace(
                    loc=mu.reshape((self.K, self.flags.batch_size, *mu.shape[1:])), scale=logvar)

        return rec_mods

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        epss = forward_results.joint_latents.epss
        losses = []
        klds = {}
        log_probs = {}
        for mod_str, subset_samples in subsets.items():
            kl_div = torch.nn.functional.kl_div(epss, subset_samples, reduce=False, log_target=True).sum(-1)
            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[mod_str].items()]

            lpx_z = torch.stack(lpx_z).sum(0)

            loss = lpx_z
            losses.append(loss)
            log_probs[mod_str] = lpx_z.mean()
            klds[mod_str] = log_mean_exp(kl_div).sum()

        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()
        print(total_loss)

        # joint_div average of all subset divs
        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))
        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds

    def conditioned_generation(self, input_samples: dict, subset_key: str, style=None):
        """
        Generate samples conditioned with input samples for a given subset.

        subset_key str: The key indicating which subset is used for the generation.
        """

        # infer latents from batch
        enc_mods, joint_latents = self.inference(input_samples)

        subset_embedding = joint_latents.subsets[subset_key].mean(dim=0)
        cond_mod_in = ReparamLatent(content=subset_embedding, style=style)
        return self.generate_from_latents(cond_mod_in)


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

    def calc_log_probs(self, rec_mods: dict, batch_d: dict) -> typing.Tuple[dict, float]:
        log_probs = {}
        weighted_log_prob = 0.0
        for mod_str, mod in self.modalities.items():
            ba = batch_d[mod_str]
            log_probs[mod_str] = -mod.calc_log_prob(out_dist=rec_mods[mod_str], target=ba,
                                                    norm_value=self.flags.batch_size)

            weighted_log_prob += mod.rec_weight * log_probs[mod.name]
        return log_probs, weighted_log_prob


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


class GfMoPVAE(MoPoEMMVae):
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


class iwmopgfm(iwMMVAE, MopGfM):
    def __init__(self, exp, flags, modalities, subsets):
        MopGfM.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)

        self.K = flags.K

    def forward(self, input_batch: dict) -> iwForwardResults:
        enc_mods, joint_latents = self.inference(input_batch)

        # reconstruct modalities
        rec_mods = self.decode(enc_mods, joint_latents)

        return iwForwardResults(enc_mods=enc_mods, joint_latents=joint_latents, rec_mods=rec_mods)

    def inference(self, input_batch) -> tuple[Mapping[str, BaseEncMod], iwJointLatents]:
        enc_mods, joint_latents = super().inference(input_batch)

        subsets = {}
        zss = {}
        for subset_str, subset in joint_latents.subsets.items():
            qz_x_tilde = distr.Normal(loc=subset.mu, scale=subset.logvar.exp())
            subsets[subset_str] = iwSubset(qz_x_tilde=qz_x_tilde, zs=qz_x_tilde.rsample(torch.Size([self.K])))

        # find the subset will all modalities to get the joint distr
        max_subset_size = max(len(subset_str.split('_')) for subset_str in joint_latents.fusion_subsets_keys)

        joint_distr = subsets[[subset_str for subset_str in joint_latents.fusion_subsets_keys if
                               len(subset_str.split('_')) == max_subset_size][0]]

        joint_latents = iwJointLatents(fusion_subsets_keys=joint_latents.fusion_subsets_keys, subsets=subsets, zss=zss,
                                       joint_distr=joint_distr)

        return enc_mods, joint_latents

    def encode(self, input_batch: Mapping[str, Tensor]) -> Mapping[str, BaseEncMod]:
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}

                _, _, class_mu, class_logvar = mod.encoder(input_batch[mod_str])

                latents_class = Distr(mu=class_mu,
                                      logvar=F.softmax(class_logvar, dim=-1) * class_logvar.size(-1) + 1e-6)
                enc_mods[mod_str] = BaseEncMod(latents_class=latents_class)

        return enc_mods

    def decode(self, enc_mods: Mapping[str, BaseEncMod], joint_latents: iwJointLatents) -> dict:
        """Decoder outputs each reconstructed modality as a dict."""
        rec_mods = {}
        for subset_str, subset in joint_latents.subsets.items():
            rec_mods[subset_str] = {}
            for out_mod_str, dec_mod in self.modalities.items():
                mu, logvar = dec_mod.decoder(None,
                                             subset.zs.reshape(
                                                 (self.K * self.flags.batch_size, self.flags.class_dim)))
                rec_mods[subset_str][out_mod_str] = distr.Laplace(
                    loc=mu.reshape((self.K, self.flags.batch_size, *mu.shape[1:])), scale=logvar)

        return rec_mods

    def calculate_loss(self, forward_results: iwForwardResults, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        losses = []
        klds = {}
        log_probs = {}
        for mod_str, subset in subsets.items():
            lpz = distr.Laplace(loc=torch.zeros(1, self.flags.class_dim, device=self.flags.device),
                                scale=torch.ones(1, self.flags.class_dim, device=self.flags.device)).log_prob(
                subset.zs).sum(-1)

            lqz_x = log_mean_exp(
                torch.stack(
                    [subset_.qz_x_tilde.log_prob(subset_.zs).sum(-1) for _, subset_ in subsets.items()]))

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[mod_str].items()]

            lpx_z = torch.stack(lpx_z).sum(0)
            kl_div = lpz - lqz_x

            loss = lpx_z + kl_div
            losses.append(loss)
            log_probs[mod_str] = lpx_z.mean()
            klds[mod_str] = log_mean_exp(kl_div).sum()

        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()

        # joint_div average of all subset divs
        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))
        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds
