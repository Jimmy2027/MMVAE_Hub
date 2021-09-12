# -*- coding: utf-8 -*-
from mmvae_hub import log
from mmvae_hub.utils.dataclasses.Dataclasses import *
from mmvae_hub.utils.metrics.average_meters import AverageMeter
from mmvae_hub.utils.utils import dict2json


class BaseCallback:
    def __init__(self, exp):
        self.exp = exp
        self.flags = exp.flags
        self.epoch_time = AverageMeter('epoch_time', precision=0)

        self.beta = self.flags.min_beta

    def update_epoch(self, train_results: BaseBatchResults, test_results: BaseTestResults, epoch: int,
                     epoch_time: float):
        self.epoch_time.update(epoch_time)

        # set beta_warmup coefficient
        self.beta = min(
            [max([self.flags.min_beta,
                  (epoch * (self.flags.max_beta - self.flags.min_beta)) / max([self.flags.beta_warmup, 1.])]),
             self.flags.max_beta])
        log.info(f'beta = {self.beta}')

        self.maybe_send_to_db(train_results=train_results, test_results=test_results, epoch=epoch, beta=self.beta)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == self.flags.end_epoch:
            self.exp.mm_vae.save_networks(epoch)

        return self.beta

    def maybe_send_to_db(self, train_results: BaseBatchResults, test_results: BaseTestResults, epoch: int, beta: float):
        """
        Send epoch results to db if use_db flags is set, otherwise save epoch results as jsonfile.
        """

        epoch_results_dict = {'train_results': {**train_results.__dict__},
                              'test_results': {**test_results.__dict__} if test_results is not None else None,
                              'epoch_time': self.epoch_time.get_average(),
                              'beta': beta}

        if self.flags.use_db == 1:
            epoch_results = self.exp.experiments_database.get_experiment_dict()['epoch_results']
            epoch_results[f'{epoch}'] = epoch_results_dict
            self.exp.experiments_database.insert_dict({'epoch_results': epoch_results})

        elif self.flags.use_db == 2:
            epoch_results_dir = self.flags.dir_experiment_run / 'epoch_results'
            epoch_results_dir.mkdir(parents=True, exist_ok=True)
            dict2json(out_path=epoch_results_dir / f'{epoch}.json', d=epoch_results_dict)
