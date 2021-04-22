import os

import torch
import torch.nn as nn

from mmvae_hub.base import BaseMMVae
from mmvae_hub.base.evaluation.divergence_measures.mm_div import calc_alphaJSD_modalities, calc_group_divergence_moe, \
    poe, calc_group_divergence_poe
from mmvae_hub.base.utils import utils


class VAEPolymnist(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)
        self.num_modalities = len(modalities.keys())
        self.flags = flags
        self.modalities = modalities
        self.subsets = subsets
        self.encoders = [modalities["m%d" % m].encoder.to(flags.device) for m in range(self.num_modalities)]
        self.decoders = [modalities["m%d" % m].decoder.to(flags.device) for m in range(self.num_modalities)]
        self.likelihoods = [modalities["m%d" % m].likelihood for m in range(self.num_modalities)]

        weights = utils.reweight_weights(torch.Tensor(flags.alpha_modalities))
        self.weights = weights.to(flags.device)
        if flags.modality_moe or flags.modality_jsd:
            self.modality_fusion = self.moe_fusion
            if flags.modality_moe:
                self.calc_joint_divergence = self.divergence_moe
            if flags.modality_jsd:
                self.calc_joint_divergence = self.divergence_jsd
        elif flags.modality_poe:
            self.modality_fusion = self.poe_fusion
            self.calc_joint_divergence = self.divergence_poe

    def forward(self, input_batch):
        latents = self.inference(input_batch)

        results = dict()
        results['latents'] = latents

        results['group_distr'] = latents['joint']
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1])
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights'])
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        enc_mods = latents['modalities']
        results_rec = dict()
        for m in range(self.num_modalities):
            x_m = input_batch['m%d' % m]
            if x_m is not None:
                style_mu, style_logvar = enc_mods['m%d_style' % m]
                if self.flags.factorized_representation:
                    style_embeddings = utils.reparameterize(mu=style_mu, logvar=style_logvar)
                else:
                    style_embeddings = None
                rec = self.likelihoods[m](*self.decoders[m](style_embeddings, class_embeddings))
                results_rec['m%d' % m] = rec
        results['rec'] = results_rec
        return results

    # def encode(self, x_m, m):
    def encode(self, input_batch):
        enc_mods = {}
        for m in range(self.num_modalities):
            x_m = input_batch['m%d' % m] if "m%d" % m in input_batch.keys() else None
            if x_m is not None:
                latents = self.encoders[m](x_m)
                latents_style = latents[:2]
                latents_class = latents[2:]
            else:
                latents_style = [None, None]
                latents_class = [None, None]
            enc_mods["m%d" % m] = latents_class
            enc_mods["m%d_style" % m] = latents_style
        return enc_mods

    def get_random_styles(self, num_samples):
        styles = dict()
        for m in range(self.num_modalities):
            if self.flags.factorized_representation:
                z_style_m = torch.randn(num_samples, self.flags.style_dim)
                z_style_m = z_style_m.to(self.flags.device)
            else:
                z_style_m = None
            styles["m%d" % m] = z_style_m
        return styles

    def get_random_style_dists(self, num_samples):
        styles = dict()
        for m in range(self.num_modalities):
            s_mu_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            s_logvar_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            dist_m = [s_mu_m, s_logvar_m]
            styles["m%d" % m] = dist_m
        return styles

    def generate_from_latents(self, latents):
        cond_gen = {}
        for m in range(self.num_modalities):
            suff_stats = self.generate_sufficient_statistics_from_latents(latents)
            cond_gen_m = suff_stats["m%d" % m].mean
            cond_gen["m%d" % m] = cond_gen_m
        return cond_gen

    def generate_sufficient_statistics_from_latents(self, latents):
        cond_gen = {}
        for m in range(self.num_modalities):
            style_m = latents['style']['m%d' % m]
            content = latents['content']
            cond_gen_m = self.likelihoods[m](*self.decoders[m](style_m, content))
            cond_gen["m%d" % m] = cond_gen_m
        return cond_gen

    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_samples = dict()
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key]
            content_rep = utils.reparameterize(mu=mu, logvar=logvar)
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents)
        return cond_gen_samples

    def cond_generation_2a(self, latent_distribution_pairs, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_2a = dict()
        for p, pair in enumerate(latent_distribution_pairs.keys()):
            ld_pair = latent_distribution_pairs[pair]
            mu_list = [];
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

    def save_networks(self):
        for m in range(self.num_modalities):
            torch.save(self.encoders[m].state_dict(), os.path.join(self.flags.dir_checkpoints, "encoderM%d" % m))
