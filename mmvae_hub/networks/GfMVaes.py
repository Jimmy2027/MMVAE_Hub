import typing

import torch
import torch.distributions as distr
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from mmvae_hub.evaluation.divergence_measures.kl_div import log_normal_diag, log_normal_standard
from mmvae_hub.evaluation.divergence_measures.mm_div import GfMMMDiv, GfMoPDiv, PGfMMMDiv, BaseMMDiv, MoFoGfMMMDiv, \
    BMoGfMMMDiv
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.MixtureVaes import MoPoEMMVae
from mmvae_hub.networks.flows.AffineFlows import AffineFlow
from mmvae_hub.networks.iwVaes import log_mean_exp, iwMMVAE
from mmvae_hub.networks.utils.utils import get_distr
from mmvae_hub.utils.Dataclasses.gfmDataclasses import SubsetMoFoGfM, JointLatentsMoGfM, \
    JointLatentsGfMoP, JointLatentsMoFoGfM
from mmvae_hub.utils.Dataclasses.iwdataclasses import *
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods, mixture_component_selection_embedding
from torch.distributions.normal import Normal


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


class iwMoGfMVAE_old(iwMMVAE, BaseMMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)
        self.mm_div = GfMMMDiv(flags=flags, K=self.K)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim,
                               nbr_coupling_block_layers=flags.nbr_coupling_block_layers)

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

        transformed_enc_mods = {
            mod_key: self.flow(
                torch.cat(tuple(self.reparam_with_eps(distr.latents_class, epss[k_idx]).unsqueeze(dim=0) for k_idx in
                                range(self.K)),
                          dim=0).reshape((self.K * batch_size, self.flags.class_dim)))[
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

        subset_samples = {k: samples.reshape((self.K, batch_size, self.flags.class_dim)) for k, samples in
                          subset_samples.items()}

        z_joint = torch.cat([v.squeeze() for _, v in subset_samples.items()], dim=0)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM(joint_embedding=joint_embedding,
                                   subsets=subset_samples,
                                   subset_samples=subset_samples, enc_mods=enc_mods, epss=epss)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        epss = forward_results.joint_latents.epss
        losses = []
        klds = {}
        log_probs = {}
        for mod_str, subset_samples in subsets.items():
            epss = torch.where(epss.abs() <= 0.001, torch.tensor(0.01, device=self.flags.device), epss)
            kl_div = (subset_samples * torch.log((subset_samples / epss).abs() + 1e-4)).mean(-1)

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[mod_str].items()]

            lpx_z = torch.stack(lpx_z).sum(0)

            loss = lpx_z + self.flags.beta * kl_div

            losses.append(loss)
            log_probs[mod_str] = lpx_z.mean()

            klds[mod_str] = self.flags.beta * log_mean_exp(kl_div).sum()

        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()

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


class BaseiwMoGfMVAE(iwMMVAE, BaseMMVAE):
    """Base class for the Importance Weighted Mixture of Generalized f-Means VAE methods"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)
        self.mm_div = GfMMMDiv(flags=flags, K=self.K)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim,
                               nbr_coupling_block_layers=flags.nbr_coupling_block_layers)
        self.prior = Normal(torch.zeros((self.flags.batch_size, self.flags.class_dim), device=self.flags.device),
                            torch.ones((self.flags.batch_size, self.flags.class_dim), device=self.flags.device))
        self.qz_x = get_distr(flags.qz_x)  # posterior

    def decode(self, enc_mods: Mapping[str, BaseEncMod], joint_latents: iwJointLatents) -> dict:
        """Decoder outputs each reconstructed modality as a dict."""
        rec_mods = {}
        for subset_str, subset in joint_latents.subsets.items():
            subset_samples = subset[0].reshape((self.K * self.flags.batch_size, self.flags.class_dim))
            rec_mods[subset_str] = {
                out_mod_str: dec_mod.calc_likelihood(
                    None, class_embeddings=subset_samples, unflatten=(self.K, self.flags.batch_size)
                )
                for out_mod_str, dec_mod in self.modalities.items()
            }
        return rec_mods

    def encode(self, input_batch: Mapping[str, Tensor]) -> Mapping[str, BaseEncMod]:
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}

                _, _, class_mu, class_logvar = mod.encoder(input_batch[mod_str])

                latents_class = Normal(class_mu, F.softmax(class_logvar, dim=-1) * class_logvar.size(-1) + 1e-6)
                enc_mods[mod_str] = BaseEncMod(latents_class=latents_class)

        return enc_mods

    def conditioned_generation(self, input_samples: dict, subset_key: str, style=None):
        """
        Generate samples conditioned with input samples for a given subset.

        subset_key str: The key indicating which subset is used for the generation.
        """

        # infer latents from batch
        enc_mods, joint_latents = self.inference(input_samples)

        subset_embedding = joint_latents.subsets[subset_key][0].mean(dim=0)
        cond_mod_in = ReparamLatent(content=subset_embedding, style=style)
        return self.generate_from_latents(cond_mod_in)

    def generate_sufficient_statistics_from_latents(self, latents: ReparamLatent) -> Mapping[str, Distribution]:
        cond_gen = {}
        for mod_str, mod in self.modalities.items():
            style_m = latents.style[mod_str]
            content = latents.content
            cond_gen_m = mod.px_z(*mod.decoder(style_m, content))
            cond_gen[mod_str] = cond_gen_m
        return cond_gen


class iwMoGfMVAE(BaseiwMoGfMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsGfM:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        subset_samples = {}

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.loc.shape[0]

        # sample K samples from enc_mod
        zmss = {
            mod_key: enc_mod.latents_class.rsample((self.K,)) for mod_key, enc_mod in enc_mods.items()}

        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(zms.reshape(self.K * batch_size, self.flags.class_dim))[0] for
                                mod_key, zms in
                                zmss.items()}

        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod.name].unsqueeze(dim=0)), dim=0)
            # mean of zks
            z_mean = torch.mean(subset_zks, dim=0)

            # calculate inverse flow
            samples = self.flow.rev(z_mean)[0]

            subset_samples[s_key] = samples

        subset_samples = {k: samples.reshape((self.K, batch_size, self.flags.class_dim)) for k, samples in
                          subset_samples.items()}

        # z_joint has nbr_subsets*K samples
        z_joint = torch.cat([v for _, v in subset_samples.items()], dim=0)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM(joint_embedding=joint_embedding,
                                   subsets=subset_samples,
                                   subset_samples=subset_samples, enc_mods=enc_mods, epss=None)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        losses = []
        klds = {}
        log_probs = {}
        # take one zm from each uni modal post., from this get a zs, compute likelihood of zs for each of the unimodal post, aggregate over this.
        for sub_str, subset_samples in subsets.items():
            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            qzs_xm = torch.cat(
                [enc_mod.latents_class.log_prob(subset_samples) for _, enc_mod in
                 forward_results.enc_mods.items()]).mean(0).sum(-1)

            kl_div = self.flags.beta * (qzs_xm - lpz)
            # print('kl_div: ', kl_div.mean(), kl_div.max())

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)

            # loss = -(lpx_z - (lqz_x - lpz))
            # loss = lpx_z - kl_div
            loss = lpx_z

            # print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()

            klds[sub_str] = log_mean_exp(kl_div).sum()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then sum over number of subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()
        # print('total loss: ', total_loss)

        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))

        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds


class iwMoGfMVAE2(BaseiwMoGfMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsiwMoGfM2:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.loc.shape[0]

        # M*k samples from the unimodal posteriors
        zmss = {mod_key: enc_mod.latents_class.rsample((self.K,)) for mod_key, enc_mod in enc_mods.items()}
        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(zms.reshape(self.K * batch_size, self.flags.class_dim)) for
                                mod_key, zms in zmss.items()}

        subset_samples = {}
        # approximations of the distributions of the sum of random variables (srv)
        srv_proxies = {}
        z_Gfs = {}
        z_joint = torch.Tensor().to(self.flags.device)
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                # subset samples are tuples: (samples, log_det_J)
                subset_samples[s_key] = (zmss[s_key], 0)
                srv_proxies[s_key] = enc_mods[s_key].latents_class
                z_Gf = zmss[s_key]
                z_joint = torch.cat([z_joint, zmss[s_key]])

            else:
                # sum of random variables
                subset_tf_enc_mods = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[s_key]])
                z_Gf = subset_tf_enc_mods.mean(dim=0)

                # calculate inverse flow
                zss, log_det_J = self.flow.rev(z_Gf)

                # approximate the sum of random variables with a gaussian
                z_Gf = z_Gf.reshape(self.K, -1, self.flags.class_dim)
                subset_tf_enc_mods = subset_tf_enc_mods.reshape(subset_tf_enc_mods.shape[0], self.K, -1,
                                                                self.flags.class_dim)
                # take the average mean over the K samples and average std
                srv_proxies[s_key] = Normal(z_Gf.mean(0), subset_tf_enc_mods.std(0).mean(0))

                subset_samples[s_key] = (zss.reshape((self.K, batch_size, self.flags.class_dim)),
                                         log_det_J.reshape((self.K, batch_size)))
                z_joint = torch.cat([z_joint, subset_samples[s_key][0]])

            z_Gfs[s_key] = z_Gf

        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM2(joint_embedding=joint_embedding,
                                    z_Gfs=z_Gfs,
                                    subsets=subset_samples, subset_samples=subset_samples,
                                    enc_mods=enc_mods, srv_proxies=srv_proxies)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        enc_mods = forward_results.enc_mods
        losses = []
        klds = {}
        log_probs = {}

        for sub_str, subset_samples in subsets.items():
            # distribution of sum of random variables inside f-mean
            Gf = forward_results.joint_latents.srv_proxies[sub_str]
            z_Gf = forward_results.joint_latents.z_Gfs[sub_str]
            # print('z_Gf: ', z_Gf.mean())

            lqz_x = Gf.log_prob(z_Gf).sum(-1) + subset_samples[1]
            # print('lqz_x: ', lqz_x.mean())
            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            d_kl = lqz_x - lpz

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)

            # loss = -(lpx_z - (lqz_x - lpz))
            if self.flags.beta == 0:
                loss = lpx_z
            else:
                loss = lpx_z - self.flags.beta * d_kl
            # print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()
            klds[sub_str] = d_kl.mean()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then sum over number of subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()
        # print('total loss: ', total_loss)

        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))

        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds


class iwMoGfMVAE4(BaseiwMoGfMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE where the joint distribution is assumed to be gaussian with inferred mean ind std."""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsiwMoGfM2:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.loc.shape[0]

        # M*k samples from the unimodal posteriors
        zmss = {mod_key: enc_mod.latents_class.rsample((self.K,)) for mod_key, enc_mod in enc_mods.items()}
        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(zms.reshape(self.K * batch_size, self.flags.class_dim)) for
                                mod_key, zms in zmss.items()}

        subset_samples = {}
        # approximations of the distributions of the sum of random variables (srv)
        srv_proxies = {}
        z_Gfs = {}
        z_joint = torch.Tensor().to(self.flags.device)
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                # subset samples are tuples: (samples, log_det_J)
                subset_samples[s_key] = (zmss[s_key], 0)
                srv_proxies[s_key] = enc_mods[s_key].latents_class
                z_Gf = zmss[s_key]
                z_joint = torch.cat([z_joint, zmss[s_key]])

            else:
                # sum of random variables
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[s_key]]).mean(dim=0)

                # calculate inverse flow
                zss, log_det_J = self.flow.rev(z_Gf)

                # approximate the sum of random variables with a gaussian
                z_Gf = z_Gf.reshape(self.K, -1, self.flags.class_dim)
                srv_proxies[s_key] = Normal(z_Gf.mean(0), z_Gf.std(0))

                subset_samples[s_key] = (zss.reshape((self.K, batch_size, self.flags.class_dim)),
                                         log_det_J.reshape((self.K, batch_size)))
                z_joint = torch.cat([z_joint, subset_samples[s_key][0]])

            z_Gfs[s_key] = z_Gf

        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM2(joint_embedding=joint_embedding,
                                    z_Gfs=z_Gfs,
                                    subsets=subset_samples, subset_samples=subset_samples,
                                    enc_mods=enc_mods, srv_proxies=srv_proxies)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        enc_mods = forward_results.enc_mods
        losses = []
        klds = {}
        log_probs = {}

        for sub_str, subset_samples in subsets.items():
            subset_posterior = Normal(loc=subset_samples[0].mean(0), scale=subset_samples[0].std(0),
                                      validate_args=False)

            lqz_x = subset_posterior.log_prob(subset_samples[0]).sum(-1) + subset_samples[1]
            # print('lqz_x: ', lqz_x.mean())
            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            d_kl = lqz_x - lpz

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)

            # loss = -(lpx_z - (lqz_x - lpz))
            if self.flags.beta == 0:
                loss = lpx_z
            else:
                loss = lpx_z - self.flags.beta * d_kl
            # print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()
            klds[sub_str] = d_kl.mean()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then sum over number of subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum()
        # print('total loss: ', total_loss)

        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))

        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds


class iwMoGfMVAE_amortized(BaseiwMoGfMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsiwMoGfMVAE_amortized:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.loc.shape[0]

        # M*k samples from the unimodal posteriors
        zmss = {mod_key: enc_mod.latents_class.rsample((self.K,)) for mod_key, enc_mod in enc_mods.items()}
        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(zms.reshape(self.K * batch_size, self.flags.class_dim))
                                for mod_key, zms in zmss.items()}

        priors_tf_enc_mods = {
            mod_key: Normal(self.flow(enc_mod.latents_class.loc)[0],
                            torch.ones(self.flags.class_dim, device=self.flags.device))
            for mod_key, enc_mod in enc_mods.items()
        }

        subset_samples = {}
        z_joint = torch.Tensor().to(self.flags.device)
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                # subset samples are tuples: (samples, log_det_J)
                subset_samples[s_key] = (zmss[s_key], 0)
                z_joint = torch.cat([z_joint, zmss[s_key]])

            else:
                # sum of random variables
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[s_key]]).mean(dim=0)

                # calculate inverse flow
                zss, log_det_J = self.flow.rev(z_Gf)

                subset_samples[s_key] = (zss.reshape((self.K, batch_size, self.flags.class_dim)),
                                         log_det_J.reshape((self.K, batch_size)))

        # reshape the transformed_enc_mods
        transformed_enc_mods = {
            k: (
                samples[0].reshape((self.K, batch_size, self.flags.class_dim)),
                samples[1].reshape((self.K, batch_size)))
            for
            k, samples in
            transformed_enc_mods.items()}

        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfMVAE_amortized(joint_embedding=joint_embedding,
                                                transformed_enc_mods=transformed_enc_mods,
                                                subsets=subset_samples, subset_samples=subset_samples,
                                                enc_mods=enc_mods, zmss=zmss, priors_tf_enc_mods=priors_tf_enc_mods)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        enc_mods = forward_results.enc_mods
        transformed_enc_mods = forward_results.joint_latents.transformed_enc_mods
        zmss = forward_results.joint_latents.zmss
        losses = []
        klds = {}
        log_probs = {}
        # amortized_priors = torch.stack([Normal(transformed_enc_mod,
        #              torch.ones(self.flags.batch_size, self.flags.class_dim, device=self.flags.device)) for transformed_enc_mod in ])
        # minimize divergence between f(zm) and amortized prior
        interm_loss = torch.stack([(
                enc_mods[key].latents_class.log_prob(zmss[key]).sum(-1)
                - forward_results.joint_latents.priors_tf_enc_mods[key].log_prob(transformed_enc_mods[key][0]).sum(-1)
                - transformed_enc_mods[key][1]
        ) for key in enc_mods]).sum(0)

        for sub_str, subset_samples in subsets.items():
            # subset size is the number of modalities included in the subset
            subset_size = len(self.subsets[sub_str])

            if subset_size == 1:
                Gf = enc_mods[sub_str].latents_class
                z_Gf = zmss[sub_str]

            else:
                # distribution of sum of random variables inside f-mean
                Gf = Normal(
                    torch.stack(
                        [transformed_enc_mod[0].mean(0) for _, transformed_enc_mod in transformed_enc_mods.items()]
                    ).mean(0),
                    torch.tensor(1 / subset_size).sqrt() * torch.ones(self.flags.batch_size, self.flags.class_dim,
                                                                      device=self.flags.device)
                )
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[sub_str]]).mean(dim=0)

            # print('z_Gf: ', z_Gf.mean())
            lqz_x = Gf.log_prob(z_Gf).sum(-1) + subset_samples[1]
            # print('lqz_x: ', lqz_x.mean())

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)
            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            d_kl = lqz_x - lpz
            # loss = -(lpx_z - (lqz_x - lpz))
            if self.flags.beta == 0:
                loss = lpx_z
            else:
                loss = lpx_z - self.flags.beta * d_kl
            # print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()
            klds[sub_str] = d_kl.mean()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then sum over number of subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum() + 2.0 * interm_loss.sum()
        # print('total loss: ', total_loss)

        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))

        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds


class iwMoGfMVAE_multiloss(BaseiwMoGfMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE with an additional loss function that pushes the transformed enc mods to be a std normal distr"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsiwMoGfMVAE_amortized:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.loc.shape[0]

        # M*k samples from the unimodal posteriors
        zmss = {mod_key: enc_mod.latents_class.rsample((self.K,)) for mod_key, enc_mod in enc_mods.items()}
        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(zms.reshape(self.K * batch_size, self.flags.class_dim))
                                for mod_key, zms in zmss.items()}

        subset_samples = {}
        z_joint = torch.Tensor().to(self.flags.device)
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                # subset samples are tuples: (samples, log_det_J)
                subset_samples[s_key] = (zmss[s_key], 0)
                z_joint = torch.cat([z_joint, zmss[s_key]])

            else:
                # sum of random variables
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[s_key]]).mean(dim=0)

                # calculate inverse flow
                zss, log_det_J = self.flow.rev(z_Gf)

                subset_samples[s_key] = (zss.reshape((self.K, batch_size, self.flags.class_dim)),
                                         log_det_J.reshape((self.K, batch_size)))

        # reshape the transformed_enc_mods
        transformed_enc_mods = {
            k: (
                samples[0].reshape((self.K, batch_size, self.flags.class_dim)),
                samples[1].reshape((self.K, batch_size)))
            for
            k, samples in
            transformed_enc_mods.items()}

        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfMVAE_amortized(joint_embedding=joint_embedding,
                                                transformed_enc_mods=transformed_enc_mods,
                                                subsets=subset_samples, subset_samples=subset_samples,
                                                enc_mods=enc_mods, zmss=zmss, priors_tf_enc_mods=None)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        enc_mods = forward_results.enc_mods
        transformed_enc_mods = forward_results.joint_latents.transformed_enc_mods
        zmss = forward_results.joint_latents.zmss
        losses = []
        klds = {}
        log_probs = {}

        # minimize divergence between f(zm) and amortized prior
        interm_loss = torch.stack([(
                enc_mods[key].latents_class.log_prob(zmss[key]).sum(-1)
                - self.prior.log_prob(transformed_enc_mods[key][0]).sum(-1)
                - transformed_enc_mods[key][1]
        ) for key in enc_mods]).sum(0)

        for sub_str, subset_samples in subsets.items():
            # subset size is the number of modalities included in the subset
            subset_size = len(self.subsets[sub_str])

            if subset_size == 1:
                Gf = enc_mods[sub_str].latents_class
                z_Gf = zmss[sub_str]

            else:
                # distribution of sum of random variables inside f-mean
                Gf = Normal(torch.zeros(self.flags.batch_size, self.flags.class_dim, device=self.flags.device),
                            torch.tensor(1 / subset_size).sqrt() * torch.ones(self.flags.batch_size,
                                                                              self.flags.class_dim,
                                                                              device=self.flags.device)
                            )
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[sub_str]]).mean(dim=0)

            # print('z_Gf: ', z_Gf.mean())
            lqz_x = Gf.log_prob(z_Gf).sum(-1) + subset_samples[1]
            # print('lqz_x: ', lqz_x.mean())

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)
            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            d_kl = lqz_x - lpz
            # loss = -(lpx_z - (lqz_x - lpz))
            if self.flags.beta == 0:
                loss = lpx_z
            else:
                loss = lpx_z - self.flags.beta * d_kl

            # print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()
            klds[sub_str] = d_kl.mean()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then sum over number of subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum() + 2.0 * interm_loss.sum()
        # print('total loss: ', total_loss)

        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))

        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds


class iwMoGfMVAE_multiloss_(BaseiwMoGfMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE with an additional loss function that pushes the transformed enc mods to be a std normal distr"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsiwMoGfMVAE_amortized:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.loc.shape[0]

        # M*k samples from the unimodal posteriors
        zmss = {mod_key: enc_mod.latents_class.rsample((self.K,)) for mod_key, enc_mod in enc_mods.items()}
        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(zms.reshape(self.K * batch_size, self.flags.class_dim))
                                for mod_key, zms in zmss.items()}

        subset_samples = {}
        z_joint = torch.Tensor().to(self.flags.device)
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                # subset samples are tuples: (samples, log_det_J)
                subset_samples[s_key] = (zmss[s_key], 0)
                z_joint = torch.cat([z_joint, zmss[s_key]])

            else:
                # sum of random variables
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[s_key]]).mean(dim=0)

                # calculate inverse flow
                zss, log_det_J = self.flow.rev(z_Gf)

                subset_samples[s_key] = (zss.reshape((self.K, batch_size, self.flags.class_dim)),
                                         log_det_J.reshape((self.K, batch_size)))

        # reshape the transformed_enc_mods
        transformed_enc_mods = {
            k: (
                samples[0].reshape((self.K, batch_size, self.flags.class_dim)),
                samples[1].reshape((self.K, batch_size)))
            for
            k, samples in
            transformed_enc_mods.items()}

        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfMVAE_amortized(joint_embedding=joint_embedding,
                                                transformed_enc_mods=transformed_enc_mods,
                                                subsets=subset_samples, subset_samples=subset_samples,
                                                enc_mods=enc_mods, zmss=zmss, priors_tf_enc_mods=None)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        enc_mods = forward_results.enc_mods
        transformed_enc_mods = forward_results.joint_latents.transformed_enc_mods
        zmss = forward_results.joint_latents.zmss
        losses = []
        klds = {}
        log_probs = {}

        # minimize divergence between f(zm) and amortized prior
        interm_loss = torch.stack([(
                enc_mods[key].latents_class.log_prob(zmss[key]).sum(-1)
                - self.prior.log_prob(transformed_enc_mods[key][0]).sum(-1)
                - transformed_enc_mods[key][1]
        ) for key in enc_mods]).sum(0)

        for sub_str, subset_samples in subsets.items():
            # subset size is the number of modalities included in the subset
            subset_size = len(self.subsets[sub_str])

            if subset_size == 1:
                Gf = enc_mods[sub_str].latents_class
                z_Gf = zmss[sub_str]

            else:
                # distribution of sum of random variables inside f-mean
                Gf = Normal(torch.zeros(self.flags.batch_size, self.flags.class_dim, device=self.flags.device),
                            torch.tensor(1 / subset_size).sqrt() * torch.ones(self.flags.batch_size,
                                                                              self.flags.class_dim,
                                                                              device=self.flags.device)
                            )
                z_Gf = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[sub_str]]).mean(dim=0)

            # print('z_Gf: ', z_Gf.mean())
            lqz_x = Gf.log_prob(z_Gf).sum(-1) + subset_samples[1]
            # print('lqz_x: ', lqz_x.mean())

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)
            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            d_kl = lpz - lqz_x
            # loss = -(lpx_z - (lqz_x - lpz))
            if self.flags.beta == 0:
                loss = lpx_z
            else:
                loss = lpx_z - self.flags.beta * d_kl

            # print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()
            klds[sub_str] = d_kl.mean()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then sum over number of subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).sum() + 2.0 * interm_loss.sum()
        # print('total loss: ', total_loss)

        joint_div = torch.cat(tuple(div.unsqueeze(dim=0) for _, div in klds.items()))

        # normalize with the number of samples
        joint_div = joint_div.mean()
        return total_loss, joint_div, log_probs, klds


class iwMoGfMVAE3(iwMMVAE, BaseMMVAE):
    """Importance Weighted Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        BaseMMVAE.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)
        self.mm_div = GfMMMDiv(flags=flags, K=self.K)
        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim,
                               nbr_coupling_block_layers=flags.nbr_coupling_block_layers)

        self.prior = Normal(torch.zeros(self.flags.class_dim, device=self.flags.device),
                            torch.ones(self.flags.class_dim, device=self.flags.device))

    def reparam_with_eps(self, distr: Distr, eps: Tensor):
        """Apply the reparameterization trick on distr with given epsilon"""
        std = distr.logvar.mul(0.5).exp_()
        # print(std.max())
        return eps.mul(std).add_(distr.mu)

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsiwMoGfM2:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].latents_class.mu.shape[0]

        transformed_enc_mods = {
            mod_key: self.flow(
                torch.cat(tuple(distr.latents_class.reparameterize().unsqueeze(dim=0) for _ in
                                range(self.K)),
                          dim=0).reshape((self.K * batch_size, self.flags.class_dim))) for mod_key, distr in
            enc_mods.items()}

        subset_samples = {}
        for s_key in batch_subsets:
            subset_zks = torch.Tensor().to(self.flags.device)
            for mod in self.subsets[s_key]:
                subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod.name][0].unsqueeze(dim=0)), dim=0)
            # mean of zks
            z_mean = torch.mean(subset_zks, dim=0)

            # calculate inverse flow
            samples = self.flow.rev(z_mean)

            subset_samples[s_key] = samples

        subset_samples = {k: (
            samples[0].reshape((self.K, batch_size, self.flags.class_dim)), samples[1].reshape((self.K, batch_size)))
            for
            k, samples in
            subset_samples.items()}

        # z_joint has nbr_subsets*K samples
        z_joint = torch.cat([v[0] for _, v in subset_samples.items()], dim=0)
        joint_embedding = JointEmbeddingFoEM(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsiwMoGfM2(joint_embedding=joint_embedding, transformed_enc_mods=transformed_enc_mods,
                                    subsets=subset_samples,
                                    subset_samples=subset_samples, enc_mods=enc_mods)

    def calculate_loss(self, forward_results, batch_d: dict) -> tuple[float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        transformed_enc_mods = forward_results.joint_latents.transformed_enc_mods
        enc_mods = forward_results.enc_mods
        losses = []
        klds = {}
        log_probs = {}

        logprobs_tf_encmods = {
            mod_k: Normal(enc_mods[mod_k].latents_class.mu, enc_mods[mod_k].latents_class.logvar.exp()).log_prob(
                transformed_enc_mod[0].reshape(self.K, self.flags.batch_size, self.flags.class_dim)).sum(-1) -
                   transformed_enc_mod[1].reshape(self.K, self.flags.batch_size) for mod_k, transformed_enc_mod in
            transformed_enc_mods.items()}

        for sub_str, subset_samples in subsets.items():
            lMz_x = torch.stack(
                [logprobs_tf_encmods[mod_k] for mod_k in sub_str.split('_')]).mean(0)
            lqz_x = lMz_x + subset_samples[1]

            lpz = self.prior.log_prob(subset_samples[0]).sum(-1)

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[sub_str].items()]

            # sum over #mods in subset
            lpx_z = torch.stack(lpx_z).sum(0)
            d_kl = lqz_x - lpz
            # loss = -(lpx_z - (lqz_x - lpz))
            loss = lpx_z - self.flags.beta * d_kl
            print('dkl: ', d_kl.mean())
            print('lMz_x: ', lMz_x)
            print('lpx_z: ', lpx_z)
            print('lqz_x: ', lqz_x)
            print('lpz: ', lpz)
            print('loss: ', loss.mean())
            losses.append(loss)
            log_probs[sub_str] = lpx_z.mean()
            klds[sub_str] = lpx_z.sum()

        # concat over k samples (get k*number of subsets) as last dim
        # take log_mean_exp over batch size
        # log_mean_exp over k, then mean over subsets
        total_loss = -log_mean_exp(torch.cat(losses, 1)).mean()
        print('total loss: ', total_loss)

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

        subset_embedding = joint_latents.subsets[subset_key][0].mean(dim=0)
        cond_mod_in = ReparamLatent(content=subset_embedding, style=style)
        return self.generate_from_latents(cond_mod_in)

    def generate_sufficient_statistics_from_latents(self, latents: ReparamLatent) -> Mapping[str, Distribution]:
        cond_gen = {}
        for mod_str, mod in self.modalities.items():
            style_m = latents.style[mod_str]
            content = latents.content
            cond_gen_m = mod.likelihood(*mod.decoder(style_m, content))
            cond_gen[mod_str] = cond_gen_m
        return cond_gen


class MoGfMVAE(iwMoGfMVAE):
    """Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        iwMoGfMVAE.__init__(self, exp, flags, modalities, subsets)
        self.K = 1


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

        self.flow = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim,
                               nbr_coupling_block_layers=flags.nbr_coupling_block_layers)

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
        self.flow_mus = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim,
                                   nbr_coupling_block_layers=flags.nbr_coupling_block_layers)
        self.flow_logvars = AffineFlow(flags.class_dim, flags.num_gfm_flows, coupling_dim=flags.coupling_dim,
                                       nbr_coupling_block_layers=flags.nbr_coupling_block_layers)

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


class iwmopgfm(iwMMVAE, PGfMVAE):
    def __init__(self, exp, flags, modalities, subsets):
        PGfMVAE.__init__(self, exp, flags, modalities, subsets)
        iwMMVAE.__init__(self, flags)
        self.mm_div = BaseMMDiv()
        self.prior = get_distr(flags.prior)(loc=torch.zeros(1, self.flags.class_dim, device=self.flags.device),
                                            scale=torch.ones(1, self.flags.class_dim, device=self.flags.device))
        self.qz_x = get_distr(flags.qz_x)  # posterior
        self.K = flags.K

    def encode(self, input_batch: Mapping[str, Tensor]) -> Mapping[str, BaseEncMod]:
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}

                _, _, class_mu, class_logvar = mod.encoder(input_batch[mod_str])

                latents_class = self.qz_x(class_mu, F.softmax(class_logvar, dim=-1) * class_logvar.size(-1) + 1e-6)
                enc_mods[mod_str] = BaseEncMod(latents_class=latents_class)

        return enc_mods

    def forward(self, input_batch: dict) -> iwForwardResults:
        enc_mods, joint_latents = self.inference(input_batch)

        # reconstruct modalities
        rec_mods = self.decode(enc_mods, joint_latents)

        return iwForwardResults(enc_mods=enc_mods, joint_latents=joint_latents, rec_mods=rec_mods)

    def inference(self, input_batch) -> tuple[Mapping[str, BaseEncMod], iwJointLatents]:
        enc_mods, joint_latents = super().inference(input_batch)

        zss = {}
        subsets = {
            subset_str: iwSubset(
                qz_x_tilde=subset, zs=subset.rsample(torch.Size([self.K]))
            )
            for subset_str, subset in joint_latents.subsets.items()
        }

        # find the subset will all modalities to get the joint distr
        max_subset_size = max(len(subset_str.split('_')) for subset_str in joint_latents.fusion_subsets_keys)

        joint_distr = subsets[[subset_str for subset_str in joint_latents.fusion_subsets_keys if
                               len(subset_str.split('_')) == max_subset_size][0]]

        joint_latents = iwJointLatents(fusion_subsets_keys=joint_latents.fusion_subsets_keys, subsets=subsets, zss=zss,
                                       joint_distr=joint_distr)

        return enc_mods, joint_latents

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
            # for the unimodal subset, the subset distr is equal to the uni modal distr
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

    def calculate_loss(self, forward_results: iwForwardResults, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        losses = []
        klds = {}
        log_probs = {}
        for mod_str, subset in subsets.items():
            lpz = self.prior.log_prob(subset.zs).sum(-1)

            lqz_x = log_mean_exp(
                torch.stack(
                    [subset_.qz_x_tilde.log_prob(subset_.zs).sum(-1) for _, subset_ in subsets.items()]
                )
            )
            #
            # lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
            #          for out_mod_str, px_z in forward_results.rec_mods[mod_str].items()]

            lpx_z = [self.modalities[out_mod_str].log_likelihood(px_z, batch_d[out_mod_str]).view(*px_z.batch_shape[:2],
                                                                                                  -1).sum(-1)
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

    def gfm(self, distrs: typing.List[Distr]) -> Distr:
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)

        for distr in distrs:
            mus = torch.cat((mus, distr.mu.unsqueeze(dim=0)), dim=0)
            logvars = torch.cat((logvars, distr.logvar.unsqueeze(dim=0)), dim=0)

        mu_average = torch.mean(mus, dim=0)
        logvar_average = torch.mean(logvars, dim=0)
        scale = self.flow_logvars.rev(logvar_average)[0]
        ok = Normal.arg_constraints["scale"].check(scale)
        bad_elements = scale[~ok]

        return self.qz_x(loc=self.flow_mus.rev(mu_average)[0], scale=F.softmax(scale, dim=-1) * scale.size(-1) + 1e-6)

    def encode_expert(self, expert_distr: Distr) -> Distr:
        mu_k, _ = self.flow_mus(expert_distr.loc)
        logvar_k, _ = self.flow_logvars(expert_distr.scale)
        return Distr(mu=mu_k, logvar=logvar_k)


class MopGfM(iwmopgfm):
    """Mixture of parameter GfM method."""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)
        self.K = 1  # mopfgm = iwmopgfm with K = 1
