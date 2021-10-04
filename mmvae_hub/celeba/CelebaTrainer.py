# -*- coding: utf-8 -*-
from mmvae_hub.utils.utils import save_and_log_flags
from torch.utils.tensorboard import SummaryWriter

from mmvae_hub.base.BaseCallback import BaseCallback
from mmvae_hub.base.BaseTrainer import BaseTrainer
from mmvae_hub.celeba.CelebaLogger import CelebALogger


class CelebaTrainer(BaseTrainer):
    def __init__(self, exp):
        super(CelebaTrainer, self).__init__(exp)

    def _set_callback(self):
        return BaseCallback(self.exp)

    def _setup_tblogger(self):
        writer = SummaryWriter(self.flags.dir_logs)
        tb_logger = CelebALogger(self.flags.str_experiment, writer)
        str_flags = save_and_log_flags(self.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        return tb_logger
