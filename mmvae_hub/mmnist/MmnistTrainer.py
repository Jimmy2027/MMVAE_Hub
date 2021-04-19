# -*- coding: utf-8 -*-
from mmvae_hub.base import BaseTrainer
from mmvae_hub.mmnist import MMNISTExperiment


class MmnistTrainer(BaseTrainer):
    def __init__(self, exp: MMNISTExperiment):
        super(MmnistTrainer, self).__init__(exp)
