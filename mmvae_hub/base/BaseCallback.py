# -*- coding: utf-8 -*-

from mmvae_hub.base.utils.Dataclasses import *
from mmvae_hub.base.utils.average_meters import AverageMeter


class BaseCallback:
    def __init__(self, exp):
        self.exp = exp
        self.flags = exp.flags
        self.epoch_time = AverageMeter('epoch_time', precision=0)

    def update_epoch(self, train_results: BaseBatchResults, test_results: BaseTestResults, epoch: int,
                     epoch_time: float):
        self.epoch_time.update(epoch_time)

        self.maybe_send_to_db(train_results=train_results, test_results=test_results, epoch=epoch)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == self.flags.end_epoch:
            self.exp.mm_vae.save_networks(epoch)

    def maybe_send_to_db(self, train_results: BaseBatchResults, test_results: BaseTestResults, epoch: int):
        """
        Send epoch results to db if use_db flags is set.
        """
        if self.flags.use_db:
            epoch_results = self.exp.experiments_database.get_experiment_dict()['epoch_results']
            epoch_results[f'{epoch}'] = {'train_results': {**train_results.__dict__},
                                         'test_results': {**test_results.__dict__},
                                         'epoch_time': self.epoch_time.get_average()}
            self.exp.experiments_database.insert_dict({'epoch_results': epoch_results})
