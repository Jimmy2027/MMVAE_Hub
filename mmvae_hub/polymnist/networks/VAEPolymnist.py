import torch
import torch.nn as nn

from mmvae_hub.networks.BaseMMVae import BaseMMVae_
from mmvae_hub.evaluation.divergence_measures.mm_div import poe
from mmvae_hub.base.utils import utils


class VAEPolymnist(BaseMMVae_, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)
        self.num_modalities = len(modalities.keys())
        self.flags = flags
        self.modalities = modalities
        self.subsets = subsets

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
        results = {'latents': latents, 'group_distr': latents['joint']}
        div = self.calc_joint_divergence(latents['mus'], latents['logvars'], latents['weights'])
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        enc_mods = latents['modalities']
        results_rec = {}
        class_embeddings = utils.reparameterize(latents['joint'][0], latents['joint'][1])
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch.keys():
                style_mu, style_logvar = enc_mods[mod_str + "_style"]
                if self.flags.factorized_representation:
                    style_embeddings = utils.reparameterize(mu=style_mu, logvar=style_logvar)
                else:
                    style_embeddings = None
                rec = mod.likelihood(*mod.decoder(style_embeddings, class_embeddings))
                results_rec[mod_str] = rec
        results['rec'] = results_rec
        return results

    # def encode(self, x_m, m):
    def encode(self, input_batch):
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch.keys():
                x_m = input_batch[mod_str]
                latents = mod.encoder(x_m)
                latents_style = latents[:2]
                latents_class = latents[2:]
            else:
                latents_style = [None, None]
                latents_class = [None, None]
            enc_mods[mod_str] = latents_class
            enc_mods[mod_str + "_style"] = latents_style
        return enc_mods

    def get_random_styles(self, num_samples):
        styles = {}
        for mod_str in self.modalities:
            if self.flags.factorized_representation:
                z_style_m = torch.randn(num_samples, self.flags.style_dim)
                z_style_m = z_style_m.to(self.flags.device)
            else:
                z_style_m = None
            styles[mod_str] = z_style_m
        return styles

    def get_random_style_dists(self, num_samples):
        styles = {}
        for mod_str in self.modalities:
            s_mu_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            s_logvar_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            dist_m = [s_mu_m, s_logvar_m]
            styles[mod_str] = dist_m
        return styles

    def generate_from_latents(self, latents):
        cond_gen = {}
        for mod_str in self.modalities:
            suff_stats = self.generate_sufficient_statistics_from_latents(latents)
            cond_gen_m = suff_stats[mod_str].mean
            cond_gen[mod_str] = cond_gen_m
        return cond_gen

    def generate_sufficient_statistics_from_latents(self, latents):
        cond_gen = {}
        for mod_str, mod in self.modalities.items():
            style_m = latents['style'][mod_str]
            content = latents['content']
            cond_gen_m = mod.likelihood(*mod.decoder(style_m, content))
            cond_gen[mod_str] = cond_gen_m
        return cond_gen

    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_samples = {}
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
