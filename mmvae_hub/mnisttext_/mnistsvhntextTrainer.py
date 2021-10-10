# -*- coding: utf-8 -*-
from mmvae_hub.base.BaseCallback import BaseCallback
from mmvae_hub.base.BaseTrainer import BaseTrainer


class mnistsvhnTrainer(BaseTrainer):
    def __init__(self, exp):
        super(mnistsvhnTrainer, self).__init__(exp)

    def _set_callback(self):
        return BaseCallback(self.exp)
