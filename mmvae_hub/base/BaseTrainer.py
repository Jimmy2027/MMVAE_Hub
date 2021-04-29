# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmvae_hub import log
from mmvae_hub.base import BaseExperiment
from mmvae_hub.base.evaluation.eval_metrics.coherence import test_generation, flatten_cond_gen_values
from mmvae_hub.base.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mmvae_hub.base.evaluation.eval_metrics.representation import test_clf_lr_all_subsets
from mmvae_hub.base.evaluation.eval_metrics.representation import train_clf_lr_all_subsets
from mmvae_hub.base.evaluation.eval_metrics.sample_quality import calc_prd_score
from mmvae_hub.base.evaluation.losses import calc_log_probs, calc_klds, calc_klds_style, calc_style_kld
from mmvae_hub.base.experiment_vis.utils import run_notebook_convert
from mmvae_hub.base.utils import BaseTBLogger
from mmvae_hub.base.utils.average_meters import *
from mmvae_hub.base.utils.Dataclasses import *
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.base.utils.utils import save_and_log_flags, at_most_n, get_items_from_dict


class BaseTrainer:
    def __init__(self, exp: BaseExperiment):
        self.exp = exp
        self.flags = exp.flags
        self.tb_logger = self._setup_tblogger()
        self.callback = self._set_callback()

    def _setup_tblogger(self):
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = BaseTBLogger(self.flags.str_experiment, writer)
        str_flags = save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        return tb_logger

    @abstractmethod
    def _set_callback(self):
        pass

    def run_epochs(self):
        test_results = None
        for epoch in tqdm(range(self.flags.start_epoch, self.flags.end_epoch), postfix='epochs'):
            # one epoch of training and testing
            self.train()
            test_results = self.test(epoch)
            self.callback.update_epoch(epoch)

        self.finalize()
        return test_results

    def train(self):
        model = self.exp.mm_vae.train()

        d_loader = DataLoader(self.exp.dataset_train, batch_size=self.flags.batch_size, shuffle=True,
                              num_workers=self.flags.dataloader_workers,
                              drop_last=True)
        training_steps = self.flags.steps_per_training_epoch

        average_meters = {
            'total_loss': AverageMeter('total_test_loss'),
            'klds': AverageMeterDict('klds'),
            'log_probs': AverageMeterDict('log_probs'),
            'joint_divergence': AverageMeter('joint_divergence'),
            'latents': AverageMeterLatents('latents', self.flags.factorized_representation),
        }

        for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
            batch_d = model.batch_to_device(batch_d)

            # forward pass
            forward_results: BaseForwardResults = model(batch_d)

            # calculate the loss
            total_loss, joint_divergence, log_probs, klds = model.calculate_loss(forward_results, batch_d)

            # backprop
            self.exp.optimizer.zero_grad()
            total_loss.backward()
            self.exp.optimizer.step()
            results = {**forward_results.__dict__, 'joint_divergence': joint_divergence}

            batch_results = {
                'total_loss': total_loss.item(),
                'klds': get_items_from_dict(klds),
                'log_probs': get_items_from_dict(log_probs),
                'joint_divergence': results['joint_divergence'].item(),
                'latents': forward_results.enc_mods,
            }

            for key, value in batch_results.items():
                average_meters[key].update(value)
        train_results = {k: v.get_average() for k, v in average_meters.items()}
        self.tb_logger.write_training_logs(**train_results)

    def test(self, epoch) -> dict:
        with torch.no_grad():
            model = self.exp.mm_vae.eval()

            d_loader = DataLoader(self.exp.dataset_test, batch_size=self.flags.batch_size, shuffle=True,
                                  num_workers=self.flags.dataloader_workers, drop_last=True)
            training_steps = self.flags.steps_per_training_epoch

            average_meters = {
                'total_loss': AverageMeter('total_test_loss'),
                'klds': AverageMeterDict('klds'),
                'log_probs': AverageMeterDict('log_probs'),
                'joint_divergence': AverageMeter('joint_divergence'),
                'latents': AverageMeterLatents('latents', self.flags.factorized_representation),
            }

            for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
                batch_d = model.batch_to_device(batch_d)
                forward_results: BaseForwardResults = model(batch_d)
                # calculate the loss
                total_loss, joint_divergence, log_probs, klds = model.calculate_loss(forward_results, batch_d)
                results = {**forward_results.__dict__, 'joint_divergence': joint_divergence}

                batch_results = {
                    'total_loss': total_loss.item(),
                    'klds': get_items_from_dict(klds),
                    'log_probs': get_items_from_dict(log_probs),
                    'joint_divergence': results['joint_divergence'].item(),
                    'latents': forward_results.enc_mods,
                }

                for key in batch_results:
                    average_meters[key].update(batch_results[key])

            test_results = {k: v.get_average() for k, v in average_meters.items()}
            self.tb_logger.write_testing_logs(**test_results)

            log.info('generating plots')
            plots = generate_plots(self.exp, epoch)
            self.tb_logger.write_plots(plots, epoch)

            if (epoch + 1) % self.flags.eval_freq == 0 or (epoch + 1) == self.flags.end_epoch:
                if self.flags.eval_lr:
                    log.info('evaluation of latent representation')
                    clf_lr = train_clf_lr_all_subsets(self.exp)
                    lr_eval = test_clf_lr_all_subsets(clf_lr, self.exp)
                    self.tb_logger.write_lr_eval(lr_eval)
                    test_results['lr_eval'] = lr_eval

                if self.flags.use_clf:
                    log.info('test generation')
                    gen_eval = test_generation(self.exp)
                    self.tb_logger.write_coherence_logs(gen_eval)
                    test_results['gen_eval'] = flatten_cond_gen_values(gen_eval)

                if self.flags.calc_nll:
                    log.info('estimating likelihoods')
                    lhoods = estimate_likelihoods(self.exp)
                    self.tb_logger.write_lhood_logs(lhoods)
                    test_results['lhoods'] = lhoods

                if self.flags.calc_prd and ((epoch + 1) % self.flags.eval_freq_fid == 0):
                    log.info('calculating prediction score')
                    prd_scores = calc_prd_score(self.exp)
                    self.tb_logger.write_prd_scores(prd_scores)
                    test_results['prd_scores'] = prd_scores

        return {k: v for k, v in test_results.items() if k in ['total_loss', 'lr_eval', 'text_gen_eval']}

    def finalize(self):
        run_notebook_convert(self.flags.dir_experiment_run)


class BaseTrainer_:
    def __init__(self, exp: BaseExperiment):
        self.exp = exp
        self.flags = exp.flags
        self.tb_logger = self._setup_tblogger()
        self.callback = self._set_callback()

    def _setup_tblogger(self):
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = BaseTBLogger(self.flags.str_experiment, writer)
        str_flags = save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        return tb_logger

    @abstractmethod
    def _set_callback(self):
        pass

    def run_epochs(self):
        for epoch in tqdm(range(self.flags.start_epoch, self.flags.end_epoch), postfix='epochs'):
            # one epoch of training and testing
            self.train()
            self.test(epoch)
            self.callback.update_epoch(epoch)

        self.finalize()

    def train(self):
        self.exp.mm_vae.train()

        d_loader = DataLoader(self.exp.dataset_train, batch_size=self.flags.batch_size, shuffle=True,
                              num_workers=self.flags.dataloader_workers,
                              drop_last=True)
        training_steps = self.flags.steps_per_training_epoch

        for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
            basic_routine = self.basic_routine_epoch(batch_d)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            # backprop
            self.exp.optimizer.zero_grad()
            total_loss.backward()
            self.exp.optimizer.step()
            self.tb_logger.write_training_logs(results, total_loss, log_probs, klds)

    def basic_routine_epoch(self, batch_d):
        # set up weights
        beta_style = self.flags.beta_style
        beta_content = self.flags.beta_content
        beta = self.flags.beta
        mm_vae = self.exp.mm_vae
        mods = self.exp.modalities
        total_loss = None
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = Variable(batch_d[m_key]).to(self.flags.device)
        results = mm_vae(batch_d)

        log_probs, weighted_log_prob = calc_log_probs(self.exp, results, batch_d)
        group_divergence = results['joint_divergence']

        klds = calc_klds(self.exp, results)
        if self.flags.factorized_representation:
            klds_style = calc_klds_style(self.exp, results)

        if (self.flags.modality_jsd or self.flags.modality_moe
                or self.flags.joint_elbo):
            if self.flags.factorized_representation:
                kld_style = calc_style_kld(self.exp, klds_style)
            else:
                kld_style = 0.0
            kld_content = group_divergence
            kld_weighted = beta_style * kld_style + beta_content * kld_content
            rec_weight = 1.0

            total_loss = rec_weight * weighted_log_prob + beta * kld_weighted
        elif self.flags.modality_poe:
            klds_joint = {'content': group_divergence,
                          'style': dict()}
            elbos = {}
            for m, m_key in enumerate(mods.keys()):
                mod = mods[m_key]
                if self.flags.factorized_representation:
                    kld_style_m = klds_style[m_key + '_style']
                else:
                    kld_style_m = 0.0
                klds_joint['style'][m_key] = kld_style_m
                if self.flags.poe_unimodal_elbos:
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      self.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = self.model.calc_elbo(self.exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
            elbo_joint = self.model.calc_elbo(self.exp, 'joint', log_probs, klds_joint)
            elbos['joint'] = elbo_joint
            total_loss = sum(elbos.values())

        return {
            'results': results,
            'log_probs': log_probs,
            'total_loss': total_loss,
            'klds': klds,
        }

    def test(self, epoch):
        with torch.no_grad():
            self.exp.mm_vae.eval()

            d_loader = DataLoader(self.exp.dataset_test, batch_size=self.flags.batch_size, shuffle=True,
                                  num_workers=self.flags.dataloader_workers, drop_last=True)
            training_steps = self.flags.steps_per_training_epoch

            for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
                basic_routine = self.basic_routine_epoch(batch_d)
                results = basic_routine['results']
                total_loss = basic_routine['total_loss']
                klds = basic_routine['klds']
                log_probs = basic_routine['log_probs']
                self.tb_logger.write_testing_logs(results, total_loss, log_probs, klds)

            log.info('generating plots')
            plots = generate_plots(self.exp, epoch)
            self.tb_logger.write_plots(plots, epoch)

            if (epoch + 1) % self.flags.eval_freq == 0 or (epoch + 1) == self.flags.end_epoch:
                if self.flags.eval_lr:
                    log.info('evaluation of latent representation')
                    clf_lr = train_clf_lr_all_subsets(self.exp)
                    lr_eval = test_clf_lr_all_subsets(clf_lr, self.exp)
                    self.tb_logger.write_lr_eval(lr_eval)

                if self.flags.use_clf:
                    log.info('test generation')
                    gen_eval = test_generation(self.exp)
                    self.tb_logger.write_coherence_logs(gen_eval)

                if self.flags.calc_nll:
                    log.info('estimating likelihoods')
                    lhoods = estimate_likelihoods(self.exp)
                    self.tb_logger.write_lhood_logs(lhoods)

                if self.flags.calc_prd and ((epoch + 1) % self.flags.eval_freq_fid == 0):
                    log.info('calculating prediction score')
                    prd_scores = calc_prd_score(self.exp)
                    self.tb_logger.write_prd_scores(prd_scores)

    def finalize(self):
        run_notebook_convert(self.flags.dir_experiment_run)
