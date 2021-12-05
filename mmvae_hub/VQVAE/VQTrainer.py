import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mmvae_hub import log
from mmvae_hub.VQVAE.VQTBLogger import VQTBLogger
from mmvae_hub.VQVAE.VQVAE import VQMoGFMVAE
from mmvae_hub.VQVAE.VqVaeDataclasses import VQForwardResults, VQBatchResults, VQTestResults
from mmvae_hub.base.BaseExperiment import BaseExperiment
from mmvae_hub.base.BaseTrainer import BaseTrainer
from mmvae_hub.evaluation.eval_metrics.coherence import test_generation, flatten_cond_gen_values
from mmvae_hub.evaluation.eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from mmvae_hub.evaluation.eval_metrics.sample_quality import calc_prd_score
from mmvae_hub.utils.metrics.average_meters import AverageMeter, AverageMeterDict, AverageMeterNestedDict
from mmvae_hub.utils.plotting.plotting import generate_plots
from mmvae_hub.utils.utils import at_most_n, get_items_from_dict, save_and_log_flags


class VQTrainer(BaseTrainer):
    def __init__(self, exp: BaseExperiment):
        super().__init__(exp)

    def _setup_tblogger(self):
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = VQTBLogger(self.flags.str_experiment, writer)
        str_flags = save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        return tb_logger

    def train(self):
        self.exp.set_train_mode()
        model: VQMoGFMVAE = self.exp.mm_vae

        training_steps = self.flags.steps_per_training_epoch

        d_loader, average_meters = self.setup_phase('train')

        for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
            batch_d = model.batch_to_device(batch_d)

            # forward pass
            forward_results: VQForwardResults = model(batch_d)

            # calculate the loss
            total_loss, quant_losses, rec_losses = model.calculate_loss(forward_results, batch_d)

            # backprop
            self.exp.optimizer.zero_grad()
            total_loss.backward()
            self.exp.optimizer.step()

            batch_results = {
                'total_loss': total_loss.item(),
                'quant_losses': get_items_from_dict(quant_losses),
                'rec_losses': {k: get_items_from_dict(v) for k, v in rec_losses.items()},
            }

            for key, value in batch_results.items():
                average_meters[key].update(value)

        train_results = {k: v.get_average() for k, v in average_meters.items()}
        self.tb_logger.write_basic_logs(**{k: v for k, v in train_results.items()}, phase='train')
        return VQBatchResults(**train_results)

    def test(self, epoch, last_epoch: bool) -> VQTestResults:
        with torch.no_grad():
            self.exp.set_eval_mode()
            model = self.exp.mm_vae

            training_steps = self.flags.steps_per_training_epoch
            d_loader, average_meters = self.setup_phase('test')

            for iteration, (batch_d, _) in enumerate(at_most_n(d_loader, training_steps)):
                batch_d = model.batch_to_device(batch_d)
                forward_results: VQForwardResults = model(batch_d)

                # calculate the loss
                total_loss, quant_losses, rec_losses = model.calculate_loss(forward_results, batch_d)

                batch_results = {
                    'total_loss': total_loss.item(),
                    'quant_losses': get_items_from_dict(quant_losses),
                    'rec_losses': {k: get_items_from_dict(v) for k, v in rec_losses.items()},
                }

                for key in batch_results:
                    average_meters[key].update(batch_results[key])

            averages = {k: v.get_average() for k, v in average_meters.items()}

            self.tb_logger.write_basic_logs(**{k: v for k, v in batch_results.items()}, phase='test')

            test_results = VQTestResults(**averages)

            log.info('generating plots')
            plots = generate_plots(self.exp, epoch)
            self.tb_logger.write_plots(plots, epoch)

            if self.flags.eval_lr:
                # todo
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

        average_meters = {
            'total_loss': AverageMeter(f'total_{phase}_loss'),
            'quant_losses': AverageMeterDict('quant_losses'),
            'rec_losses': AverageMeterNestedDict(name='rec_losses',
                                                 structure={k1: {k2: [] for k2 in self.exp.modalities} for k1 in
                                                            self.exp.subsets}),
        }
        return d_loader, average_meters
