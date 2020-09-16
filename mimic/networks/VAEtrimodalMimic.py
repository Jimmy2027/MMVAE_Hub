
import os

import torch
import torch.nn as nn

from mimic.networks.ConvNetworksImgMimic import EncoderImg, DecoderImg
from mimic.networks.ConvNetworksTextMimic import EncoderText, DecoderText

from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_poe
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import calc_kl_divergence
from divergence_measures.mm_div import poe

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEtrimodalMimic(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super(VAEtrimodalMimic, self).__init__(flags, modalities, subsets)
        self.encoder_pa = modalities['PA'].encoder;
        self.encoder_lat = modalities['Lateral'].encoder;
        self.encoder_text = modalities['text'].encoder;
        self.decoder_pa = modalities['PA'].decoder;
        self.decoder_lat = modalities['Lateral'].decoder;
        self.decoder_text = modalities['text'].decoder;
        self.encoder_pa = self.encoder_pa.to(flags.device);
        self.encoder_lat = self.encoder_lat.to(flags.device);
        self.encoder_text = self.encoder_text.to(flags.device);
        self.decoder_pa = self.decoder_pa.to(flags.device);
        self.decoder_lat = self.decoder_lat.to(flags.device);
        self.decoder_text = self.decoder_text.to(flags.device);
        self.lhood_pa = modalities['PA'].likelihood;
        self.lhood_lat = modalities['Lateral'].likelihood;
        self.lhood_text = modalities['text'].likelihood;


    def forward(self, input_batch):
        latents = self.inference(input_batch);
        results = dict();
        results['latents'] = latents;
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights']);
        results['group_distr'] = latents['joint'];
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);

        for k, key in enumerate(div.keys()):
            results[key] = div[key];

        input_m1 = input_batch['PA'];
        input_m2 = input_batch['Lateral'];
        input_m3 = input_batch['text'];
        results_rec = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            input_mod = input_batch[m_key];
            if input_mod is not None:
                if self.flags.factorized_representation:
                    s_mu, s_logvar = latents[m_key + '_style'];
                    s_emb = utils.reparameterize(mu=s_mu, logvar=s_logvar);
                else:
                    s_emb = None;
                if m_key == 'PA':
                    rec = self.lhood_pa(*self.decoder_pa(s_emb, class_embeddings));
                elif m_key == 'Lateral':
                    rec = self.lhood_lat(*self.decoder_lat(s_emb, class_embeddings));
                    lhood = self.lhood_lat;
                elif m_key == 'text':
                    rec = self.lhood_text(*self.decoder_text(s_emb, class_embeddings));
                results_rec[m_key] = rec;
        results['rec'] = results_rec;
        return results;


    def encode(self, input_batch):
        latents = dict();
        if 'PA' in input_batch.keys():
            i_m1 = input_batch['PA'];
            latents['PA'] = self.encoder_pa(i_m1)
            latents['PA_style'] = latents['PA'][:2]
            latents['PA'] = latents['PA'][2:]
        else:
            latents['PA_style'] = [None, None]
            latents['PA'] = [None, None]
        if 'Lateral' in input_batch.keys():
            i_m2 = input_batch['Lateral'];
            latents['Lateral'] = self.encoder_lat(i_m2)
            latents['Lateral_style'] = latents['Lateral'][:2]
            latents['Lateral'] = latents['Lateral'][2:]
        else:
            latents['Lateral_style'] = [None, None]
            latents['Lateral'] = [None, None]
        if 'text' in input_batch.keys():
            i_m3 = input_batch['text'];
            latents['text'] = self.encoder_text(i_m3)
            latents['text_style'] = latents['text'][:2]
            latents['text'] = latents['text'][2:]
        else:
            latents['text_style'] = [None, None]
            latents['text'] = [None, None]
        return latents;


    def get_random_styles(self, num_samples):
        if self.flags.factorized_representation:
            z_style_1 = torch.randn(num_samples, self.flags.style_pa_dim);
            z_style_2 = torch.randn(num_samples, self.flags.style_lat_dim);
            z_style_3 = torch.randn(num_samples, self.flags.style_text_dim);
            z_style_1 = z_style_1.to(self.flags.device);
            z_style_2 = z_style_2.to(self.flags.device);
            z_style_3 = z_style_3.to(self.flags.device);
        else:
            z_style_1 = None;
            z_style_2 = None;
            z_style_3 = None;
        styles = {'PA': z_style_1, 'Lateral': z_style_2, 'text': z_style_3};
        return styles;


    def get_random_style_dists(self, num_samples):
        s1_mu = torch.zeros(num_samples,
                            self.flags.style_pa_dim).to(self.flags.device)
        s1_logvar = torch.zeros(num_samples,
                                self.flags.style_pa_dim).to(self.flags.device);
        s2_mu = torch.zeros(num_samples,
                            self.flags.style_lat).to(self.flags.device)
        s2_logvar = torch.zeros(num_samples,
                                self.flags.style_lat_dim).to(self.flags.device);
        s3_mu = torch.zeros(num_samples,
                            self.flags.style_text_dim).to(self.flags.device)
        s3_logvar = torch.zeros(num_samples,
                                self.flags.style_text_dim).to(self.flags.device);
        m1_dist = [s1_mu, s1_logvar];
        m2_dist = [s2_mu, s2_logvar];
        m3_dist = [s3_mu, s3_logvar];
        styles = {'PA': m1_dist, 'Lateral': m2_dist, 'text': m3_dist};
        return styles;


    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;
        z_class = torch.randn(num_samples, self.flags.class_dim);
        z_class = z_class.to(self.flags.device);

        style_latents = self.get_random_styles(num_samples);
        random_latents = {'content': z_class, 'style': style_latents};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen_pa = suff_stats['PA'].mean;
        cond_gen_lat = suff_stats['Lateral'].mean;
        cond_gen_text = suff_stats['text'].mean;
        cond_gen = {'PA': cond_gen_pa,
                    'Lateral': cond_gen_lat,
                    'text': cond_gen_text};
        return cond_gen;


    def generate_sufficient_statistics_from_latents(self, latents):
        style_pa = latents['style']['PA'];
        style_lat = latents['style']['Lateral'];
        style_text = latents['style']['text'];
        content = latents['content']
        cond_gen_m1 = self.lhood_pa(*self.decoder_pa(style_pa, content));
        cond_gen_m2 = self.lhood_lat(*self.decoder_lat(style_lat, content));
        cond_gen_m3 = self.lhood_text(*self.decoder_text(style_text, content));
        return {'PA': cond_gen_m1, 'Lateral': cond_gen_m2, 'text': cond_gen_m3}


    def save_networks(self):
        torch.save(self.encoder_pa.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_pa.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_lat.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_lat.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))
        torch.save(self.encoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m3))
        torch.save(self.decoder_text.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m3))
