# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch
from mmvae_hub.base import log, BaseExperiment
from mmvae_hub.base.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.base.evaluation.eval_metrics.likelihood import estimate_likelihoods
from mmvae_hub.base.evaluation.eval_metrics.representation import test_clf_lr_all_subsets
from mmvae_hub.base.evaluation.eval_metrics.representation import train_clf_lr_all_subsets
from mmvae_hub.base.evaluation.eval_metrics.sample_quality import calc_prd_score
from mmvae_hub.base.evaluation.losses import calc_log_probs, calc_klds, calc_klds_style, calc_style_kld
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.base.utils.utils import save_and_log_flags, at_most_n
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mmvae_hub.base.utils import BaseTBLogger




class BaseTrainer:
    def __init__(self, exp:BaseExperiment):
        self.exp = exp
        self.flags = exp.flags
        self.tb_logger =self.setup_tblogger()
        self.model = None


    def setup_tblogger(self):
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = BaseTBLogger(self.flags.str_experiment, writer)
        str_flags = save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        return tb_logger

    def run_epochs(self):
        for epoch in tqdm(range(self.flags.start_epoch, self.flags.end_epoch), postfix='epochs'):
            # one epoch of training and testing
            self.train(exp, tb_logger)
            test(epoch, exp, tb_logger)
            # save checkpoints after every 5 epochs
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.flags.end_epoch:
                dir_network_epoch = os.path.join(self.flags.dir_checkpoints, str(epoch).zfill(4))
                if not os.path.exists(dir_network_epoch):
                    os.makedirs(dir_network_epoch)
                exp.mm_vae.save_networks()
                torch.save(exp.mm_vae.state_dict(), os.path.join(dir_network_epoch, self.flags.mm_vae_save))

    def train(self):
        mm_vae = self.model
        mm_vae.train()
        exp.mm_vae = mm_vae

        d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size, shuffle=True,
                              num_workers=exp.flags.dataloader_workers,
                              drop_last=True)
        training_steps = exp.flags.steps_per_training_epoch

        for iteration, batch in tqdm(enumerate(at_most_n(d_loader, training_steps)),
                                     total=training_steps or len(d_loader), postfix='train'):
            basic_routine = basic_routine_epoch(exp, batch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            # backprop
            exp.optimizer.zero_grad()
            total_loss.backward()
            exp.optimizer.step()
            tb_logger.write_training_logs(results, total_loss, log_probs, klds)

