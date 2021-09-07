# -*- coding: utf-8 -*-
import shutil
import time
from abc import abstractmethod

import optuna
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mmvae_hub import log
from mmvae_hub.base import BaseCallback
from mmvae_hub.base import BaseExperiment
from mmvae_hub.evaluation.eval_metrics.coherence import test_generation, flatten_cond_gen_values
from mmvae_hub.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mmvae_hub.evaluation.eval_metrics.representation import test_clf_lr_all_subsets
from mmvae_hub.evaluation.eval_metrics.representation import train_clf_lr_all_subsets
from mmvae_hub.evaluation.eval_metrics.sample_quality import calc_prd_score
from mmvae_hub.experiment_vis.utils import run_notebook_convert
from mmvae_hub.hyperopt.hyperopt_metrics import get_hyperopt_score
from mmvae_hub.networks.FlowVaes import FlowVAE
from mmvae_hub.utils.BaseTBLogger import BaseTBLogger
from mmvae_hub.utils.metrics.average_meters import *
from mmvae_hub.utils.plotting.plotting import generate_plots
from mmvae_hub.utils.utils import save_and_log_flags, at_most_n, get_items_from_dict, dict2json


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
            last_epoch: bool = (epoch + 1) == self.flags.end_epoch
            if (epoch + 1) % self.flags.eval_freq == 0 or last_epoch:
                test_results = self.test(epoch, last_epoch=last_epoch)

            self.exp.mm_vae.flags.beta = self.callback.update_epoch(train_results, test_results, epoch,
                                                                    time.time() - end)

            if self.flags.optuna and ((epoch + 1) % self.flags.eval_freq == 0 or (epoch + 1) == self.flags.end_epoch):
                self.flags.optuna, hyperopt_score = get_hyperopt_score(test_results=test_results,
                                                                       use_zk=isinstance(self.exp.mm_vae, FlowVAE),
                                                                       optuna_trial=self.flags.optuna)
                self.flags.optuna.report(hyperopt_score, epoch)
                # Handle pruning based on the intermediate value.
                if self.flags.optuna.should_prune():
                    raise optuna.exceptions.TrialPruned()
                test_results.hyperopt_score = hyperopt_score

        self.finalize(test_results, epoch, average_epoch_time=self.callback.epoch_time.get_average())
        return test_results

    def train(self):
        self.exp.set_train_mode()
        model = self.exp.mm_vae

        d_loader, training_steps, average_meters = self.setup_phase('train')

        for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
            batch_d = model.batch_to_device(batch_d)

            # forward pass
            forward_results: BaseForwardResults = model(batch_d)
            # calculate the loss
            total_loss, joint_divergence, log_probs, klds = model.calculate_loss(forward_results, batch_d)

            # backprop
            self.exp.optimizer.zero_grad()
            total_loss.backward()
            # temp
            for _, mod in model.modalities.items():
                # torch.nn.utils.clip_grad_norm_(mod.encoder.parameters(), 5)
                # with torch.no_grad():
                #     for p in mod.encoder.parameters():
                #         torch.nan_to_num_(p, nan=0,posinf=0, neginf=0)

                print(max(p.grad.max() for p in mod.encoder.parameters()))
                assert not sum(p.grad.isnan().all() for p in mod.encoder.parameters())

            # temp
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.exp.optimizer.step()

            # temp
            # for _, mod in model.modalities.items():
            #     assert not sum(p.isnan().all() for p in mod.encoder.parameters())
            results = {**forward_results.__dict__, 'joint_divergence': joint_divergence}

            batch_results = {
                'total_loss': total_loss.item(),
                'klds': get_items_from_dict(klds),
                'log_probs': get_items_from_dict(log_probs),
                'joint_divergence': results['joint_divergence'].item(),
                'latents': forward_results.enc_mods,
                # 'joint_latents': forward_results.joint_latents
            }

            for key, value in batch_results.items():
                average_meters[key].update(value)

        train_results = {k: v.get_average() for k, v in average_meters.items()}
        self.tb_logger.write_training_logs(**{k: v for k, v in train_results.items() if k != 'joint_latents'})
        return BaseBatchResults(**train_results)

    def test(self, epoch, last_epoch: bool) -> BaseTestResults:
        with torch.no_grad():
            self.exp.set_eval_mode()
            model = self.exp.mm_vae

            d_loader, training_steps, average_meters = self.setup_phase('test')

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
                    # 'joint_latents': forward_results.joint_latents
                }

                for key in batch_results:
                    average_meters[key].update(batch_results[key])

            averages = {k: v.get_average() for k, v in average_meters.items()}

            self.tb_logger.write_testing_logs(**{k: v for k, v in averages.items() if k != 'joint_latents'})

            test_results = BaseTestResults(joint_div=averages['joint_divergence'], **averages)

            log.info('generating plots')
            plots = generate_plots(self.exp, epoch)
            self.tb_logger.write_plots(plots, epoch)

            if self.flags.eval_lr:
                log.info('evaluation of latent representation')
                # train linear classifiers
                clf_lr_q0, clf_lr_zk = train_clf_lr_all_subsets(self.exp)

                # test linear classifiers
                # methods where the flow is applied on each modality don't have a q0.
                lr_eval_q0 = test_clf_lr_all_subsets(clf_lr_q0, self.exp, which_lr='q0') \
                    if clf_lr_q0 else None
                lr_eval_zk = test_clf_lr_all_subsets(clf_lr_zk, self.exp, which_lr='zk') \
                    if clf_lr_zk else None

                # log results
                lr_eval_results = {'q0': lr_eval_q0, 'zk': lr_eval_zk}
                log.info(f'Lr eval results: {lr_eval_results}')
                self.tb_logger.write_lr_eval(lr_eval_results)
                test_results.lr_eval_q0 = lr_eval_q0
                test_results.lr_eval_zk = lr_eval_zk

            if self.flags.use_clf:
                log.info('test generation')
                gen_eval = test_generation(self.exp)
                log.info(f'Gen eval results: {gen_eval}')
                self.tb_logger.write_coherence_logs(gen_eval)
                test_results.gen_eval = flatten_cond_gen_values(gen_eval)

            if self.flags.calc_nll:
                log.info('estimating likelihoods')
                lhoods = estimate_likelihoods(self.exp)
                self.tb_logger.write_lhood_logs(lhoods)
                test_results.lhoods = lhoods

            if self.flags.calc_prd and (((epoch + 1) % self.flags.eval_freq_fid == 0) or last_epoch):
                log.info('calculating prediction score')
                prd_scores = calc_prd_score(self.exp)
                self.tb_logger.write_prd_scores(prd_scores)
                test_results.prd_scores = prd_scores
        return test_results

    def setup_phase(self, phase: str):
        """Setup for train or test phase."""
        dataset = getattr(self.exp, f'dataset_{phase}')
        d_loader = DataLoader(dataset, batch_size=self.flags.batch_size, shuffle=True,
                              num_workers=self.flags.dataloader_workers, drop_last=True)

        training_steps = self.flags.steps_per_training_epoch

        average_meters = {
            'total_loss': AverageMeter('total_test_loss'),
            'klds': AverageMeterDict('klds'),
            'log_probs': AverageMeterDict('log_probs'),
            'joint_divergence': AverageMeter('joint_divergence'),
            'latents': AverageMeterLatents('latents', self.flags.factorized_representation),
            # 'joint_latents': AverageMeterJointLatents(model=self.exp.mm_vae, name='joint_latents',
            #                                           factorized_representation=self.flags.factorized_representation)
        }
        return d_loader, training_steps, average_meters

    def finalize(self, test_results: BaseTestResults, epoch: int, average_epoch_time):
        log.info('Finalizing.')
        # write results as json to experiment folder
        run_metadata = {'end_epoch': epoch, 'experiment_duration': time.time() - self.begin_time,
                        'mean_epoch_time': self.callback.epoch_time.get_average()}

        dict2json(self.flags.dir_experiment_run / 'results.json', test_results.__dict__ | run_metadata)

        if self.flags.use_db == 1:
            self.exp.experiments_database.insert_dict(run_metadata)
            self.exp.experiments_database.save_networks_to_db(dir_checkpoints=self.flags.dir_checkpoints, epoch=epoch,
                                                              modalities=self.exp.mm_vae.modalities)
            # self.exp.experiments_database.upload_logfile(self.flags.log_file)
            self.exp.experiments_database.upload_tensorbardlogs(self.flags.dir_experiment_run / 'logs')

            # run jupyter notebook with visualisations
            pdf_path = run_notebook_convert(self.flags.dir_experiment_run)

        # send alert
        if self.flags.norby and self.flags.dataset != 'toy':
            import ppb
            import norby
            expvis_url = ppb.upload(pdf_path, plain=True)
            self.exp.experiments_database.insert_dict({'expvis_url': expvis_url})
            norby.send_msg(f'Experiment {self.flags.experiment_uid} has finished. The experiment visualisation can be '
                           f'found here: {expvis_url}')

        if self.flags.log_file.exists():
            shutil.move(self.flags.log_file, self.flags.dir_experiment_run)
