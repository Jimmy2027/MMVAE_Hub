# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch
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
from mmvae_hub.base.experiment_vis.utils import run_notebook_convert
from mmvae_hub.base.utils import BaseTBLogger
from mmvae_hub.base.utils.average_meters import *
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
