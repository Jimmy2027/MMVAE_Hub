# -*- coding: utf-8 -*-
from mmvae_hub.base import BaseTrainer


# from mmvae_hub.polymnist import PolymnistExperiment


class PolymnistTrainer(BaseTrainer):
    # def __init__(self, exp: PolymnistExperiment):
    def __init__(self, exp):
        super(PolymnistTrainer, self).__init__(exp)
