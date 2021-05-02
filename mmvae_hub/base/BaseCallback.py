# -*- coding: utf-8 -*-
import os

from mmvae_hub.base.utils.Dataclasses import *
from mmvae_hub.base.utils.average_meters import AverageMeter


class BaseCallback:
    def __init__(self, exp):
        self.exp = exp
        self.flags = exp.flags
        self.epoch_time = AverageMeter('epoch_time', precision=0)

    def update_epoch(self, test_results: BaseTestResults, epoch: int, epoch_time: float):
        self.epoch_time.update(epoch_time)

        self.maybe_send_to_db(test_results=test_results, epoch=epoch)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == self.flags.end_epoch:

            self.exp.mm_vae.save_networks(epoch)

    def maybe_send_to_db(self, test_results: BaseTestResults, epoch: int):
        """Send to db if use_db flags is set."""
        if self.flags.use_db:
            epoch_results = {str(epoch): {**test_results.__dict__, 'epoch_time': self.epoch_time.get_average()}}
            self.exp.experiments_database.insert_dict(epoch_results)
