# -*- coding: utf-8 -*-

import shutil
import time
from abc import abstractmethod
from pathlib import Path

import nbformat
import torch
from nbconvert import HTMLExporter, PDFExporter
from nbconvert.preprocessors import ExecutePreprocessor
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
from mmvae_hub.base.utils.utils import dict2json
from mmvae_hub.base.utils.utils import save_and_log_flags, at_most_n, get_items_from_dict


class BaseTrainer:
    def __init__(self, exp: BaseExperiment):
        self.exp = exp
        self.flags = exp.flags
        self.tb_logger = self._setup_tblogger()
        self.callback: BaseCallback = self._set_callback()

        self.begin_time = time.time()

        if self.flags.kl_annealing:
            self.exp.mm_vae.flags.beta = 0

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

            self.exp.mm_vae.flags.beta = self.callback.update_epoch(train_results, test_results, epoch,
                                                                    time.time() - end)

        self.finalize(test_results, epoch, average_epoch_time=self.callback.epoch_time.get_average())
        return test_results

    def train(self):
        model = self.exp.mm_vae.train()

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
            self.exp.optimizer.step()
            results = {**forward_results.__dict__, 'joint_divergence': joint_divergence}

            batch_results = {
                'total_loss': total_loss.item(),
                'klds': get_items_from_dict(klds),
                'log_probs': get_items_from_dict(log_probs),
                'joint_divergence': results['joint_divergence'].item(),
                'latents': forward_results.enc_mods,
                'joint_latents': forward_results.joint_latents
            }

            for key, value in batch_results.items():
                average_meters[key].update(value)

        train_results = {k: v.get_average() for k, v in average_meters.items()}
        self.tb_logger.write_training_logs(**{k: v for k, v in train_results.items() if k != 'joint_latents'})
        return BaseBatchResults(**train_results)

    def test(self, epoch) -> BaseTestResults:
        with torch.no_grad():
            model = self.exp.mm_vae.eval()

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
                    'joint_latents': forward_results.joint_latents
                }

                for key in batch_results:
                    average_meters[key].update(batch_results[key])

            averages = {k: v.get_average() for k, v in average_meters.items()}

            self.tb_logger.write_testing_logs(**{k: v for k, v in averages.items() if k != 'joint_latents'})

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
            'joint_latents': AverageMeterJointLatents(method=self.flags.method, name='joint_latents',
                                                      factorized_representation=self.flags.factorized_representation)
        }
        return d_loader, training_steps, average_meters

    def finalize(self, test_results: BaseTestResults, epoch: int, average_epoch_time):
        # write results as json to experiment folder
        run_metadata = {'end_epoch': epoch, 'experiment_duration': time.time() - self.begin_time,
                        'mean_epoch_time': self.callback.epoch_time.get_average()}

        dict2json(self.flags.dir_experiment_run / 'results.json', test_results.__dict__ | run_metadata)

        if self.flags.log_file.exists():
            shutil.move(self.flags.log_file, self.flags.dir_experiment_run)

        if self.flags.use_db == 1:
            self.exp.experiments_database.insert_dict(run_metadata)
            self.exp.experiments_database.save_networks_to_db(dir_checkpoints=self.flags.dir_checkpoints, epoch=epoch,
                                                              modalities=self.exp.mm_vae.modalities)
            self.exp.experiments_database.upload_logfile(self.flags.log_file)
            self.exp.experiments_database.upload_tensorbardlogs(self.flags.dir_experiment_run / 'logs')

            # run jupyter notebook with visualisations
            pdf_path = self.run_notebook_convert(self.flags.dir_experiment_run)

        # send alert
        if self.flags.norby and self.flags.dataset != 'toy':
            import ppb
            import norby
            expvis_url = ppb.upload(pdf_path, plain=True)
            self.exp.experiments_database.insert_dict({'expvis_url': expvis_url})
            norby.send_msg(f'Experiment {self.flags.experiment_uid} has finished. The experiment visualisation can be '
                           f'found here: {expvis_url}')

    @staticmethod
    def run_notebook_convert(dir_experiment_run: Path) -> Path:
        """Run and convert the notebook to html and pdf."""

        # Copy the experiment_vis jupyter notebook to the experiment dir
        notebook_path = Path(__file__).parent.parent / 'experiment_vis/experiment_vis.ipynb'
        dest_notebook_path = dir_experiment_run / 'experiment_vis.ipynb'

        # copy notebook to experiment run
        shutil.copyfile(notebook_path, dest_notebook_path)

        log.info('Executing experiment vis notebook.')
        with open(dest_notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': str(dest_notebook_path.parent)}})

        nbconvert_path = dest_notebook_path.with_suffix('.nbconvert.ipynb')

        with open(nbconvert_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        log.info('Converting notebook to html.')
        html_path = nbconvert_path.with_suffix('.html')
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, resources) = html_exporter.from_notebook_node(nb)
        with open(html_path, 'w') as f:
            f.write(body)

        log.info('Converting notebook to pdf.')
        pdf_path = nbconvert_path.with_suffix('.pdf')
        pdf_exporter = PDFExporter()
        pdf_exporter.template_name = 'classic'
        (body, resources) = pdf_exporter.from_notebook_node(nb)
        pdf_path.write_bytes(body)

        return pdf_path
