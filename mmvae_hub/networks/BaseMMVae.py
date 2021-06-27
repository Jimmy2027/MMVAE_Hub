import os
from abc import ABC
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch.nn as nn
from torch.distributions.distribution import Distribution

from mmvae_hub.evaluation.divergence_measures.mm_div import BaseMMDiv
from mmvae_hub.evaluation.losses import calc_style_kld
from mmvae_hub.networks.utils.mixture_component_selection import mixture_component_selection as moe
from mmvae_hub.utils import utils
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.fusion_functions import *


class BaseMMVAE(ABC, nn.Module):
    def __init__(self, exp, flags, modalities, subsets):
        super(BaseMMVAE, self).__init__()
        self.exp = exp
        self.flags = flags
        self.modalities = modalities
        self.subsets = subsets
        self.metrics = None
        self.mm_div: Optional[BaseMMDiv] = None

    # ==================================================================================================================
    # Basic functions
    # ==================================================================================================================

    def forward(self, input_batch: dict) -> BaseForwardResults:
        enc_mods, joint_latents = self.inference(input_batch)
        # reconstruct modalities
        rec_mods = self.decode(enc_mods, joint_latents)

        return BaseForwardResults(enc_mods=enc_mods, joint_latents=joint_latents, rec_mods=rec_mods)

    def inference(self, input_batch) -> tuple[Mapping[str, Union[BaseEncMod, EncModPlanarMixture]], JointLatents]:
        # encode input
        enc_mods = self.encode(input_batch)
        batch_mods = [k for k in input_batch]
        # fuse latents
        joint_latent = self.fuse_modalities(enc_mods, batch_mods)

        return enc_mods, joint_latent

    def encode(self, input_batch: Mapping[str, Tensor]) -> Mapping[str, BaseEncMod]:
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}

                style_mu, style_logvar, class_mu, class_logvar, _ = mod.encoder(input_batch[mod_str])
                latents_class = Distr(mu=class_mu, logvar=class_logvar)
                enc_mods[mod_str] = BaseEncMod(latents_class=latents_class)

                if style_mu is not None:
                    latents_style = Distr(mu=style_mu, logvar=style_logvar)
                    enc_mods[mod_str].latents_style = latents_style

        return enc_mods

    def decode(self, enc_mods: Mapping[str, EncModPlanarMixture], latents_joint: JointLatents) -> dict:
        """Decoder outputs each reconstructed modality as a dict."""
        rec_mods = {}
        class_embeddings = latents_joint.get_joint_embeddings()

        for mod_str, enc_mod in enc_mods.items():
            if enc_mod.latents_style:
                latents_style = enc_mod.latents_style
                style_embeddings = latents_style.reparameterize()
            else:
                style_embeddings = None
            mod = self.modalities[mod_str]
            rec_mods[mod_str] = mod.likelihood(*mod.decoder(style_embeddings, class_embeddings))
        return rec_mods

    def calculate_loss(self, forward_results: BaseForwardResults, batch_d: dict) -> tuple[
        float, float, dict, Mapping[str, float]]:

        klds, joint_divergence = self.mm_div.calc_klds(forward_results, self.subsets,
                                                       num_samples=self.flags.batch_size,
                                                       joint_keys=getattr(forward_results.joint_latents,
                                                                          [attr for attr in
                                                                           forward_results.joint_latents.__dict__ if
                                                                           attr.startswith('joint_')][0]).mod_strs
                                                       )

        log_probs, weighted_log_prob = self.calc_log_probs(forward_results.rec_mods, batch_d)
        beta_style = self.flags.beta_style

        if self.flags.factorized_representation:
            klds_style = calc_klds_style(self.exp, forward_results.joint_latents.enc_mods)
            kld_style = calc_style_kld(self.exp, klds_style)
        else:
            kld_style = 0.0
        kld_weighted = beta_style * kld_style + self.flags.beta_content * joint_divergence
        rec_weight = 1.0
        total_loss = rec_weight * weighted_log_prob + self.flags.beta * kld_weighted

        # temp
        assert not np.isnan(total_loss.cpu().item())

        return total_loss, joint_divergence, log_probs, klds

    def calc_log_probs(self, rec_mods: dict, batch_d: dict) -> Tuple[dict, float]:
        log_probs = {}
        weighted_log_prob = 0.0
        for mod_str, mod in self.modalities.items():
            ba = batch_d[mod_str]
            log_probs[mod_str] = -mod.calc_log_prob(out_dist=rec_mods[mod_str], target=ba,
                                                    norm_value=self.flags.batch_size)

            weighted_log_prob += mod.rec_weight * log_probs[mod.name]
        return log_probs, weighted_log_prob

    # ==================================================================================================================
    # Generation
    # ==================================================================================================================

    def generate_from_latents(self, latents: ReparamLatent) -> Mapping[str, Tensor]:
        cond_gen = {}
        for mod_str in self.modalities:
            suff_stats = self.generate_sufficient_statistics_from_latents(latents)
            cond_gen_m = suff_stats[mod_str].mean
            cond_gen[mod_str] = cond_gen_m
        return cond_gen

    def generate_sufficient_statistics_from_latents(self, latents: ReparamLatent) -> Mapping[str, Distribution]:
        cond_gen = {}
        for mod_str, mod in self.modalities.items():
            style_m = latents.style[mod_str]
            content = latents.content
            cond_gen_m = mod.likelihood(*mod.decoder(style_m, content))
            cond_gen[mod_str] = cond_gen_m
        return cond_gen

    def get_random_styles(self, num_samples: int) -> Mapping[str, Optional[Tensor]]:
        styles = {}
        for mod_str in self.modalities:
            if self.flags.factorized_representation:
                z_style_m = torch.randn(num_samples, self.flags.style_dim)
                z_style_m = z_style_m.to(self.flags.device)
            else:
                z_style_m = None
            styles[mod_str] = z_style_m
        return styles

    def get_random_style_dists(self, num_samples: int) -> Mapping[str, Distr]:
        styles = {}
        for mod_str in self.modalities:
            s_mu_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            s_logvar_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            dist_m = Distr(mu=s_mu_m, logvar=s_logvar_m)
            styles[mod_str] = dist_m
        return styles

    def generate(self, num_samples=None) -> Mapping[str, Tensor]:
        """
        Generate from latents that were sampled from a normal distribution.
        """
        if num_samples is None:
            num_samples = self.flags.batch_size

        z_class = self.get_rand_samples_from_joint(num_samples)

        z_styles = self.get_random_styles(num_samples)
        random_latents = ReparamLatent(content=z_class, style=z_styles)
        return self.generate_from_latents(random_latents)

    def get_rand_samples_from_joint(self, num_samples: int):
        mu = torch.zeros(num_samples,
                         self.flags.class_dim).to(self.flags.device)
        logvar = torch.zeros(num_samples,
                             self.flags.class_dim).to(self.flags.device)
        return Distr(mu, logvar).reparameterize()

    def cond_generation(self, joint_latent: JointLatents, num_samples=None) -> Mapping[str, Mapping[str, Tensor]]:
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_samples = {}
        for key in joint_latent.subsets:
            content_rep = joint_latent.get_subset_embedding(key)
            latents = ReparamLatent(content=content_rep, style=style_latents)
            cond_gen_samples[key] = self.generate_from_latents(latents)

        joint_embedding = joint_latent.get_joint_embeddings()
        joint_latents = ReparamLatent(content=joint_embedding, style=style_latents)
        cond_gen_samples['joint'] = self.generate_from_latents(joint_latents)

        return cond_gen_samples

    def cond_generation_2a(self, latent_distribution_pairs, num_samples=None):
        # question was macht diese Funktion?
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_2a = {}
        for p, pair in enumerate(latent_distribution_pairs.keys()):
            ld_pair = latent_distribution_pairs[pair]
            mu_list = []
            logvar_list = []
            for k, key in enumerate(ld_pair['latents'].keys()):
                mu_list.append(ld_pair['latents'][key][0].unsqueeze(0))
                logvar_list.append(ld_pair['latents'][key][1].unsqueeze(0))
            mus = torch.cat(mu_list, dim=0)
            logvars = torch.cat(logvar_list, dim=0)
            # weights_pair = utils.reweight_weights(torch.Tensor(ld_pair['weights']))
            # mu_joint, logvar_joint = self.modality_fusion(mus, logvars, weights_pair)
            mu_joint, logvar_joint = poe(mus, logvars)
            c_emb = utils.reparameterize(mu_joint, logvar_joint)
            l_2a = {'content': c_emb, 'style': style_latents}
            cond_gen_2a[pair] = self.generate_from_latents(l_2a)
        return cond_gen_2a

    # ==================================================================================================================
    # Helper functions
    # ==================================================================================================================

    def fuse_modalities(self, enc_mods: Mapping[str, BaseEncMod], batch_mods: typing.Iterable[str]) -> JointLatents:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        A joint latent space is then created by fusing all subspaces.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)
        distr_subsets = {}

        # subsets that will be fused into the joint distribution
        fusion_subsets_keys = []

        # concatenate mus and logvars for every modality in each subset
        for s_key in batch_subsets:
            distr_subset = self.fuse_subset(enc_mods, s_key)
            distr_subsets[s_key] = distr_subset

            if self.fusion_condition(self.subsets[s_key], batch_mods):
                mus = torch.cat((mus, distr_subset.mu.unsqueeze(0)), dim=0)
                logvars = torch.cat((logvars, distr_subset.logvar.unsqueeze(0)), dim=0)
                fusion_subsets_keys.append(s_key)

        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(self.flags.device)
        joint_distr = self.moe_fusion(mus, logvars, weights)
        joint_distr.mod_strs = fusion_subsets_keys

        return JointLatents(fusion_subsets_keys, joint_distr=joint_distr, subsets=distr_subsets)

    def fuse_subset(self, enc_mods, s_key: str) -> Distr:
        """Fuse encoded modalities in subset."""
        mods = self.subsets[s_key]
        # fuse the subset
        mus_subset = torch.Tensor().to(self.flags.device)
        logvars_subset = torch.Tensor().to(self.flags.device)

        # get enc_mods for subset
        for mod in mods:
            latents_mod = enc_mods[mod.name].latents_class
            mus_subset = torch.cat((mus_subset, latents_mod.mu.unsqueeze(0)), dim=0)
            logvars_subset = torch.cat((logvars_subset, latents_mod.logvar.unsqueeze(0)), dim=0)

        # normalize latents by number of modalities in subset
        weights_subset = ((1 / float(len(mus_subset))) *
                          torch.ones(len(mus_subset)).to(self.flags.device))

        # fuse enc_mods
        subset_distr = self.modality_fusion(self.flags, mus_subset, logvars_subset, weights_subset)
        return subset_distr

    def moe_fusion(self, mus, logvars, weights=None) -> Distr:
        if weights is None:
            weights = self.weights

        weights = utils.reweight_weights(weights)
        return moe(self.flags, mus, logvars, weights)

    def batch_to_device(self, batch):
        """Send the batch to device as Variable."""
        return {k: Variable(v).to(self.flags.device) for k, v in batch.items()}

    def save_networks(self, epoch: int):
        dir_network_epoch = os.path.join(self.flags.dir_checkpoints, str(epoch).zfill(4))
        if not os.path.exists(dir_network_epoch):
            os.makedirs(dir_network_epoch)
        for mod_str, mod in self.modalities.items():
            torch.save(mod.encoder.state_dict(), os.path.join(dir_network_epoch, f"encoderM{mod_str}"))
            torch.save(mod.decoder.state_dict(), os.path.join(dir_network_epoch, f"decoderM{mod_str}"))

    def load_networks(self, dir_checkpoint: Path):
        for mod_str, mod in self.modalities.items():
            mod.encoder.load_state_dict(
                state_dict=torch.load(dir_checkpoint / f"encoderM{mod_str}", map_location=self.flags.device))
            mod.decoder.load_state_dict(
                state_dict=torch.load(dir_checkpoint / f"decoderM{mod_str}", map_location=self.flags.device))

    @staticmethod
    def calculate_lr_eval_scores(epoch_results: dict):
        results_dict = {}
        scores = []
        scores_lr_q0 = []
        scores_lr_zk = []

        for key, val in epoch_results['lr_eval_q0'].items():
            results_dict[f'lr_eval_q0_{key}'] = val['accuracy']
            scores_lr_q0.append(val['accuracy'])
            scores.append(val['accuracy'])
        return np.mean(scores), np.mean(scores_lr_q0), np.mean(scores_lr_zk)
