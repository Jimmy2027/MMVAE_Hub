import math

import torch
import torch.distributions as distr
import torch.nn.functional as F
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.MixtureVaes import MOEMMVae
from mmvae_hub.utils.dataclasses.Dataclasses import *
from mmvae_hub.utils.dataclasses.iwdataclasses import *
from mmvae_hub.utils.metrics.likelihood import log_mean_exp


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


class iwMoE(MOEMMVae):
    def __init__(self, exp, flags, modalities, subsets):
        MOEMMVae.__init__(self, exp, flags, modalities, subsets)

        self.K = 10

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
            qz_x_tilde = distr.Laplace(loc=subset.mu, scale=subset.logvar)
            subsets[subset_str] = iwSubset(qz_x_tilde=qz_x_tilde, zs=qz_x_tilde.rsample(torch.Size([self.K])))

        joint_latents = iwJointLatents(fusion_subsets_keys=joint_latents.fusion_subsets_keys, subsets=subsets, zss=zss)

        return enc_mods, joint_latents

    def forward_(self, input_batch: dict) -> iwForwardResults:
        enc_mods = self.inference(input_batch)
        # reconstruct modalities
        rec_mods = self.decode(enc_mods)

        return iwForwardResults(enc_mods=enc_mods, rec_mods=rec_mods)

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
        for in_mod_str, enc_mod in enc_mods.items():
            rec_mods[in_mod_str] = {}
            for out_mod_str, dec_mod in self.modalities.items():
                mu, logvar = dec_mod.decoder(None,
                                             joint_latents.subsets[in_mod_str].zs.reshape(
                                                 (self.K * self.flags.batch_size, self.flags.class_dim)))
                rec_mods[in_mod_str][out_mod_str] = distr.Laplace(
                    loc=mu.reshape((self.K, self.flags.batch_size, *mu.shape[1:])), scale=logvar)

        return rec_mods

    def calculate_loss(self, forward_results: iwForwardResults, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:
        subsets = forward_results.joint_latents.subsets
        losses = []
        klds = {}
        log_probs = {}
        for mod_str, enc_mod in forward_results.enc_mods.items():
            subset = subsets[mod_str]
            lpz = distr.Laplace(loc=torch.zeros(1, self.flags.class_dim, device=self.flags.device),
                                scale=torch.ones(1, self.flags.class_dim, device=self.flags.device)).log_prob(
                subset.zs).sum(-1)

            lqz_x = log_mean_exp(
                torch.stack(
                    [subsets[mod].qz_x_tilde.log_prob(subset.zs).sum(-1) for mod in forward_results.enc_mods]))

            lpx_z = [px_z.log_prob(batch_d[out_mod_str]).view(*px_z.batch_shape[:2], -1).sum(-1)
                     for out_mod_str, px_z in forward_results.rec_mods[mod_str].items()]

            lpx_z = torch.stack(lpx_z).sum(0)
            kl_div = lpz - lqz_x

            loss = lpx_z + kl_div
            losses.append(loss)
            log_probs[mod_str] = lpx_z.mean()
            klds[mod_str] = log_mean_exp(kl_div).sum()

        total_loss = log_mean_exp(torch.cat(losses, 1)).sum()

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

        subset_embedding = joint_latents.subsets[subset_key].qz_x_tilde.mean
        cond_mod_in = ReparamLatent(content=subset_embedding, style=style)
        return self.generate_from_latents(cond_mod_in)
