# -*- coding: utf-8 -*-

import os
import shutil
import time
from abc import abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmvae_hub import log
from mmvae_hub.base import BaseCallback
from mmvae_hub.base import BaseExperiment
from mmvae_hub.base.evaluation.eval_metrics.coherence import test_generation, flatten_cond_gen_values
from mmvae_hub.base.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mmvae_hub.base.evaluation.eval_metrics.representation import test_clf_lr_all_subsets
from mmvae_hub.base.evaluation.eval_metrics.representation import train_clf_lr_all_subsets
from mmvae_hub.base.evaluation.eval_metrics.sample_quality import calc_prd_score
from mmvae_hub.base.utils import BaseTBLogger
from mmvae_hub.base.utils.average_meters import *
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.base.utils.utils import save_and_log_flags, at_most_n, get_items_from_dict, dict2json


class BaseTrainer:
    def __init__(self, exp: BaseExperiment):
        self.exp = exp
        self.flags = exp.flags
        self.tb_logger = self._setup_tblogger()
        self.callback: BaseCallback = self._set_callback()

        self.begin_time = time.time()

    def _setup_tblogger(self):
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = BaseTBLogger(self.flags.str_experiment, writer)
        str_flags = save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        return tb_logger

    @abstractmethod
    def _set_callback(self) -> BaseCallback:
        pass

    def run_epochs(self):
        test_results = None
        end = time.time()
        for epoch in tqdm(range(self.flags.start_epoch, self.flags.end_epoch), postfix='epochs'):
            end = time.time()

            # set epoch for tb_logger
            self.tb_logger.step = epoch

            # training and testing
            train_results: BaseBatchResults = self.train()
            test_results = self.test(epoch)

            self.callback.update_epoch(train_results, test_results, epoch, time.time() - end)

        self.finalize(test_results, epoch, average_epoch_time=self.callback.epoch_time.get_average())
        return test_results

    def train(self):
        model = self.exp.mm_vae.train()

        d_loader, training_steps, average_meters = self.setup_phase()

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
        return BaseBatchResults(**train_results)

    def test(self, epoch) -> BaseTestResults:
        with torch.no_grad():
            model = self.exp.mm_vae.eval()

            d_loader, training_steps, average_meters = self.setup_phase()

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

            averages = {k: v.get_average() for k, v in average_meters.items()}
            self.tb_logger.write_testing_logs(**averages)

            test_results = BaseTestResults(joint_div=averages['joint_divergence'], **averages)

            if (epoch + 1) % self.flags.eval_freq == 0 or (epoch + 1) == self.flags.end_epoch:
                log.info('generating plots')
                plots = generate_plots(self.exp, epoch)
                self.tb_logger.write_plots(plots, epoch)

                if self.flags.eval_lr:
                    log.info('evaluation of latent representation')
                    clf_lr = train_clf_lr_all_subsets(self.exp)
                    lr_eval = test_clf_lr_all_subsets(clf_lr, self.exp)
                    self.tb_logger.write_lr_eval(lr_eval)
                    test_results.lr_eval = lr_eval

                if self.flags.use_clf:
                    log.info('test generation')
                    gen_eval = test_generation(self.exp)
                    self.tb_logger.write_coherence_logs(gen_eval)
                    test_results.gen_eval = flatten_cond_gen_values(gen_eval)

                if self.flags.calc_nll:
                    log.info('estimating likelihoods')
                    lhoods = estimate_likelihoods(self.exp)
                    self.tb_logger.write_lhood_logs(lhoods)
                    test_results.lhoods = lhoods

                if self.flags.calc_prd and ((epoch + 1) % self.flags.eval_freq_fid == 0):
                    log.info('calculating prediction score')
                    prd_scores = calc_prd_score(self.exp)
                    self.tb_logger.write_prd_scores(prd_scores)
                    test_results.prd_scores = prd_scores
        return test_results

    def setup_phase(self):
        """Setup for train or test phase."""
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
        return d_loader, training_steps, average_meters

    def finalize(self, test_results: BaseTestResults, epoch: int, average_epoch_time):
        # write results as json to experiment folder
        test_results.end_epoch = epoch
        test_results.mean_epoch_time = self.callback.epoch_time.get_average()
        test_results.experiment_duration = time.time() - self.begin_time
        dict2json(self.flags.dir_experiment_run / 'results.json', test_results.__dict__)

        # run jupyter notebook with visualisations
        self.run_notebook_convert(self.flags.dir_experiment_run)

        # todo send epoch, experiment_duration, average_epoch_time to db.

        # send alert
        if self.flags.norby and self.flags.dataset != 'toy':
            import norby
            norby.send_msg('Training has finished.')

    def run_notebook_convert(self, dir_experiment_run: Path) -> None:
        """Run and convert the notebook to html."""
        if self.flags.use_db:
            # Copy the experiment_vis jupyter notebook to the experiment dir
            notebook_path = Path(__file__).parent / 'experiment_vis/experiment_vis.ipynb'
            dest_notebook_path = dir_experiment_run / 'experiment_vis.ipynb'

            # copy notebook to experiment run
            shutil.copyfile(notebook_path, dest_notebook_path)

            nbconvert_path = dest_notebook_path.with_suffix('.nbconvert.ipynb')

            log.info('Executing experiment vis notebook.')
            os.system(f'jupyter nbconvert --to notebook --execute {dest_notebook_path}')
            log.info('Converting notebook to html.')
            os.system(f'jupyter nbconvert --to html {nbconvert_path}')

            html_path = nbconvert_path.with_suffix('.html')
            assert html_path.exists(), f'html notebook does not exist in destination {html_path}.'
