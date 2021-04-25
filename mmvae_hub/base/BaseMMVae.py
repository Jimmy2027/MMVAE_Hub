import os
import typing
from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch.nn as nn
from torch.autograd import Variable

from mmvae_hub.base.evaluation.divergence_measures.mm_div import log_normal_standard, log_normal_diag
from mmvae_hub.base.evaluation.losses import calc_klds_style, calc_style_kld, calc_klds
from mmvae_hub.base.networks import flows
from mmvae_hub.base.utils.dataclasses import *
from mmvae_hub.base.utils.fusion_functions import *


class BaseMMVAE(ABC, nn.Module):
    def __init__(self, flags, modalities):
        super(BaseMMVAE, self).__init__()
        self.flags = flags
        self.modalities = modalities
        self.metrics = None

        # self.weights = utils.reweight_weights(torch.Tensor(self.flags.alpha_modalities)).to(self.flags.device)

    @abstractmethod
    def encode(self, input_batch) -> dict:
        pass

    @abstractmethod
    def get_random_styles(self, num_samples):
        pass

    @abstractmethod
    def get_random_style_dists(self, num_samples):
        pass

    @abstractmethod
    def generate_sufficient_statistics_from_latents(self, latents):
        pass

    @abstractmethod
    def forward(self, input_batch):
        pass

    @abstractmethod
    def fuse_modalities(self, enc_mods: dict, batch_mods: typing.Iterable[str], num_samples: int):
        pass

    @abstractmethod
    def inference(self, input_batch) -> Tuple[dict, dict]:
        pass

    def batch_to_device(self, batch):
        """Send the batch to device as Variable."""
        return {k: Variable(v).to(self.flags.device) for k, v in batch.items()}

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = utils.reweight_weights(weights)
        # mus = torch.cat(mus, dim=0)
        # logvars = torch.cat(logvars, dim=0)
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags, mus, logvars, weights)
        return [mu_moe, logvar_moe]

    def calculate_loss(self, forward_results: BaseForwardResults, batch_d: dict) -> tuple[float, dict, dict, dict]:
        len(forward_results.enc_mods)
        weights = (1 / float(len(forward_results.enc_mods))) * torch.ones(len(forward_results.enc_mods)).to(
            self.flags.device)
        joint_divergences = self.calc_joint_divergence(forward_results, weights=weights)
        joint_divergence = joint_divergences['joint_div']
        log_probs, weighted_log_prob = self.calc_log_probs(forward_results.rec_mods, batch_d)
        beta_style = self.flags.beta_style
        klds = calc_klds(self.flags, forward_results.joint_latents['subsets'])

        if self.flags.factorized_representation:
            klds_style = calc_klds_style(self.exp, forward_results.joint_latents['enc_mods'])
            kld_style = calc_style_kld(self.exp, klds_style)
        else:
            kld_style = 0.0
        kld_weighted = beta_style * kld_style + self.flags.beta_content * joint_divergence
        rec_weight = 1.0

        total_loss = rec_weight * weighted_log_prob + self.flags.beta * kld_weighted

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


class SubsetFuseMMVae(BaseMMVAE, ABC):
    def __init__(self, flags, modalities, subsets):
        super(SubsetFuseMMVae, self).__init__(flags, modalities)
        self.num_modalities = len(modalities.keys())
        self.flags = flags
        self.modalities = modalities
        self.subsets = subsets
        self.metrics = None

    def forward(self, input_batch) -> BaseForwardResults:
        enc_mods, joint_latents = self.inference(input_batch)
        results_rec = {}
        joint_distr = joint_latents['joint_distr']
        class_embeddings = utils.reparameterize(joint_distr['mu'], joint_distr['logvar'])
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch.keys():
                latents_style = enc_mods[mod_str]['latents_style']
                if self.flags.factorized_representation:
                    style_embeddings = utils.reparameterize(mu=latents_style['mu'], logvar=latents_style['logvar'])
                else:
                    style_embeddings = None
                rec = mod.likelihood(*mod.decoder(style_embeddings, class_embeddings))
                results_rec[mod_str] = rec

        return BaseForwardResults(enc_mods=enc_mods, joint_latents=joint_latents, rec_mods=results_rec)

    def inference(self, input_batch) -> Tuple[dict, dict]:
        num_samples = self.flags.batch_size
        enc_mods = self.encode(input_batch)
        batch_mods = [k for k in input_batch]
        # fuse modalities
        joint_latent = self.fuse_modalities(enc_mods, batch_mods, num_samples)
        return enc_mods, joint_latent

    def fuse_modalities(self, enc_mods: dict, batch_mods: typing.Iterable[str], num_samples: int) -> dict:
        joint_latents = {}
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)
        distr_subsets = {}
        # concatenate mus and logvars for every modality in each subset
        for s_key in self.subsets:
            if s_key != '':
                mods = self.subsets[s_key]
                mus_subset = torch.Tensor().to(self.flags.device)
                logvars_subset = torch.Tensor().to(self.flags.device)
                mods_avail = True
                for mod in mods:
                    if mod.name in batch_mods:
                        latents_mod = enc_mods[mod.name]['latents_class']
                        mus_subset = torch.cat((mus_subset, latents_mod['mu'].unsqueeze(0)), dim=0)
                        logvars_subset = torch.cat((logvars_subset, latents_mod['logvar'].unsqueeze(0)), dim=0)
                    else:
                        mods_avail = False
                if mods_avail:
                    # normalize latents by number of modalities in subset
                    weights_subset = ((1 / float(len(mus_subset))) *
                                      torch.ones(len(mus_subset)).to(self.flags.device))
                    s_mu, s_logvar = self.modality_fusion(self.flags, mus_subset, logvars_subset, weights_subset)
                    distr_subsets[s_key] = [s_mu, s_logvar]
                    # fusion_condition always true
                    # store all s_mus and s_logvars in variables mus and logvars
                    if self.fusion_condition(mods, batch_mods):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)
        if self.flags.modality_jsd:
            mus = torch.cat((mus, torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device)), dim=0)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device)),
                                dim=0)
        # normalize with number of subsets
        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(self.flags.device)
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)

        joint_latents['mus'] = mus
        joint_latents['logvars'] = logvars
        joint_latents['joint_distr'] = {'mu': joint_mu, 'logvar': joint_logvar}
        joint_latents['subsets'] = distr_subsets
        return joint_latents

    def encode(self, input_batch) -> dict:
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}
                style_mu, style_logvar, class_mu, class_logvar, h = mod.encoder(input_batch[mod_str])
                latents_style = {'mu': style_mu, 'logvar': style_logvar}
                latents_class = {'mu': class_mu, 'logvar': class_logvar}

                enc_mods[mod_str]['latents_class'] = latents_class
                enc_mods[mod_str]["latents_style"] = latents_style

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

    def save_networks(self):
        for mod_str, mod in self.modalities.items():
            torch.save(mod.encoder.state_dict(), os.path.join(self.flags.dir_checkpoints, f"encoderM{mod_str}"))

    def divergence_static_prior(self, forward_results: BaseForwardResults, weights=None) -> dict:
        joint_latents = forward_results.joint_latents
        if weights is None:
            weights = self.weights
        weights = weights.clone()
        weights = utils.reweight_weights(weights)
        div_measures = calc_group_divergence_moe(self.flags, joint_latents['mus'], joint_latents['logvars'],
                                                 normalization=self.flags.batch_size)
        return {'joint_div': div_measures[0], 'individual_divs': div_measures[1]}

    def divergence_dynamic_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size)
        return {
            'joint_divergence': div_measures[0],
            'individual_divs': div_measures[1],
            'dyn_prior': div_measures[2],
        }

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = utils.reweight_weights(weights)
        # mus = torch.cat(mus, dim=0)
        # logvars = torch.cat(logvars, dim=0)
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags, mus, logvars, weights)
        return [mu_moe, logvar_moe]

    def poe_fusion(self, mus, logvars, weights=None):
        """
        Fuses all modalities in subset with product of experts method.
        """
        if self.flags.method == 'poe':
            num_samples = mus[0].shape[0]
            mus = torch.cat((mus, torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device)), dim=0)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples, self.flags.class_dim).to(self.flags.device)),
                                dim=0)
        # mus = torch.cat(mus, dim=0)
        # logvars = torch.cat(logvars, dim=0)
        mu_poe, logvar_poe = poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def fusion_condition_moe(self, subset, input_batch=None):
        return len(subset) == 1

    def fusion_condition_poe(self, subset, input_batch=None):
        return len(subset) == len(input_batch.keys())

    def fusion_condition_joint(self, subset, input_batch=None):
        return True

    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        mu = torch.zeros(num_samples,
                         self.flags.class_dim).to(self.flags.device)
        logvar = torch.zeros(num_samples,
                             self.flags.class_dim).to(self.flags.device)
        z_class = utils.reparameterize(mu, logvar)
        z_styles = self.get_random_styles(num_samples)
        random_latents = {'content': z_class, 'style': z_styles}
        return self.generate_from_latents(random_latents)

    @staticmethod
    def calc_elbo(exp, modality, recs, klds):
        flags = exp.flags
        s_weights = exp.style_weights
        kld_content = klds['content']
        if modality == 'joint':
            w_style_kld = 0.0
            w_rec = 0.0
            klds_style = klds['style']
            mods = exp.modalities
            r_weights = exp.rec_weights
            for m_key in mods:
                w_style_kld += s_weights[m_key] * klds_style[m_key]
                w_rec += r_weights[m_key] * recs[m_key]
            kld_style = w_style_kld
            rec_error = w_rec
        else:
            beta_style_mod = s_weights[modality]
            # rec_weight_mod = r_weights[modality]
            rec_weight_mod = 1.0
            kld_style = beta_style_mod * klds['style'][modality]
            rec_error = rec_weight_mod * recs[modality]
        div = flags.beta_content * kld_content + flags.beta_style * kld_style
        return rec_error + flags.beta * div


class BaseFlowMMVAE(BaseMMVAE):
    def __init__(self, flags, modalities):
        super(BaseFlowMMVAE, self).__init__(flags, modalities)
        self.flow = None

    @abstractmethod
    def encode(self, input_batch) -> dict:
        pass

    @abstractmethod
    def get_random_styles(self, num_samples):
        pass

    @abstractmethod
    def get_random_style_dists(self, num_samples):
        pass

    @abstractmethod
    def generate_sufficient_statistics_from_latents(self, latents):
        pass

    @abstractmethod
    def fuse_modalities(self, enc_mods: dict, batch_mods: typing.Iterable[str], num_samples: int):
        pass


class PlanarFlowMMVae(BaseFlowMMVAE, ABC):

    def __init__(self, flags, modalities):
        super(PlanarFlowMMVae, self).__init__(flags, modalities)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = flags.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(flags.class_dim, self.num_flows * flags.class_dim)
        self.amor_w = nn.Linear(flags.class_dim, self.num_flows * flags.class_dim)
        self.amor_b = nn.Linear(flags.class_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, input_batch: dict) -> dict:
        """
        Encoder that outputs parameters for base distribution of z and flow parameters.
        """
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}
                x_m = input_batch[mod_str]
                style_mu, style_logvar, class_mu, class_logvar, h = mod.encoder(x_m)
                latents_style = {'mu': style_mu, 'logvar': style_logvar}
                latents_class = {'mu': class_mu, 'logvar': class_logvar}
                # get amortized u an w for all flows
                enc_mods[mod_str]['flow_params'] = {
                    'u': self.amor_u(h).view(self.flags.batch_size, self.num_flows, self.flags.class_dim, 1),
                    'w': self.amor_w(h).view(self.flags.batch_size, self.num_flows, 1, self.flags.class_dim),
                    'b': self.amor_b(h).view(self.flags.batch_size, self.num_flows, 1, 1)}

                enc_mods[mod_str]['latents_class'] = latents_class
                enc_mods[mod_str]["latents_style"] = latents_style

        return enc_mods

    def apply_flows(self, enc_mods: dict) -> dict:
        """Apply the flow for each modality."""
        for mod_str, enc_mod in enc_mods.items():
            log_det_j = 0.
            latents_class = enc_mod['latents_class']
            flow_params = enc_mod['flow_params']
            # Sample z_0
            z = [utils.reparameterize(mu=latents_class['mu'], logvar=latents_class['logvar'])]

            # Normalizing flows
            for k in range(self.num_flows):
                flow_k = getattr(self, 'flow_' + str(k))
                z_k, log_det_jacobian = flow_k(z[k], flow_params['u'][:, k, :, :], flow_params['w'][:, k, :, :],
                                               flow_params['b'][:, k, :, :])
                z.append(z_k)
                log_det_j += log_det_jacobian
            enc_mods[mod_str]['z0'] = z[0]
            enc_mods[mod_str]['zk'] = z[-1]
            enc_mods[mod_str]['log_det_j'] = log_det_j
        return enc_mods

    def decode(self, enc_mods: dict, latents_joint: dict) -> dict:
        """Decoder outputs each reconstructed modality as a dict."""
        rec_mods = {}
        class_embeddings = utils.reparameterize(latents_joint['mu'], latents_joint['logvar'])
        for mod_str, enc_mod in enc_mods.items():
            if 'latents_style' in enc_mod:
                latents_style = enc_mods['latents_style']
                style_embeddings = utils.reparameterize(mu=latents_style['mu'], logvar=latents_style['logvar'])
            else:
                style_embeddings = None
            mod = self.modalities[mod_str]
            rec_mods[mod_str] = mod.likelihood(*mod.decoder(style_embeddings, class_embeddings))
        return rec_mods

    def inference(self, input_batch: dict) -> Tuple[dict, dict]:
        # get the latent for each class
        enc_mods = self.encode(input_batch)

        # pass the latents of each class through the flow
        enc_mods = self.apply_flows(enc_mods)

        # fuse modalities
        joint_latent = self.fuse_modalities(enc_mods)

        return enc_mods, joint_latent

    def forward(self, input_batch: dict) -> BaseForwardResults:
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        enc_mods, joint_latents = self.inference(input_batch)
        # reconstruct modalities
        rec_mods = self.decode(enc_mods, joint_latents)
        return BaseForwardResults(enc_mods=enc_mods, joint_latents=joint_latents, rec_mods=rec_mods)

    def calc_joint_divergence(self, enc_mods: dict) -> BaseDivergences:
        """Calculate the joint divergence as a mixture of the uni modal divergences."""
        num_mods = len(enc_mods)
        mods_div = {}
        klds = torch.Tensor()
        klds = klds.to(self.flags.device)
        # todo: check in old code what the weights are
        weights = (1 / float(num_mods)) * torch.ones(num_mods).to(self.flags.device)
        weights = utils.reweight_weights(weights).to(self.flags.device)

        for mod_idx, (mod_str, enc_mod) in enumerate(enc_mods.items()):
            latents_class = enc_mod['latents_class']
            kld_ind = self.calculate_singlemod_divergence(latents_class['mu'], latents_class['logvar'], enc_mod['zk'],
                                                          enc_mod['z0'], enc_mod['log_det_j'])
            mods_div[mod_str] = kld_ind

            klds = torch.cat((klds, torch.unsqueeze(kld_ind, dim=0)), dim=0)

        # sum over all the individual kl divergences
        joint_div = (weights * klds).sum(dim=0)
        return BaseDivergences(joint_div=joint_div, mods_div=mods_div)

    @staticmethod
    def calculate_singlemod_divergence(z_mu, z_logvar, zk, z0, log_det_j) -> Tensor:
        """Calculate the KL Divergence: DKL = E_q0[ ln q(z_0) - ln p(z_k) ] - E_q_z0[\sum_k log |det dz_k/dz_k-1| ]."""
        # ln p(z_k)  (not averaged)
        log_p_zk = log_normal_standard(zk, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_diag(z0, mean=z_mu, log_var=z_logvar, dim=1)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = torch.sum(log_q_z0 - log_p_zk)

        # sum over batches
        summed_ldj = torch.sum(log_det_j)

        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        return (summed_logs - summed_ldj)

    def fuse_modalities(self, enc_mods: dict) -> dict:
        """Get the mixture of the latents."""
        # todo take style into account
        mus = torch.Tensor().to(self.flags.device)
        logvars = torch.Tensor().to(self.flags.device)
        for mod_str, enc_mod in enc_mods.items():
            enc_mod = enc_mod['latents_class']
            mus = torch.cat((mus, enc_mod['mu'].unsqueeze(0)), dim=0)
            logvars = torch.cat((logvars, enc_mod['logvar'].unsqueeze(0)), dim=0)
        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(self.flags.device)
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)
        return {'joint_distr': {'mu': joint_mu, 'logvar': joint_logvar}}

    def get_random_styles(self, num_samples):
        pass

    def generate_sufficient_statistics_from_latents(self, latents):
        pass

    def get_random_style_dists(self, num_samples):
        pass


class JointElboMMVae(SubsetFuseMMVae):
    def __init__(self, flags, modalities, subsets):
        super(JointElboMMVae, self).__init__(flags, modalities, subsets)
        self.modality_fusion = poe_fusion
        self.fusion_condition = fusion_condition_joint
        self.calc_joint_divergence = self.divergence_static_prior
