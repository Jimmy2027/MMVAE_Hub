# -*- coding: utf-8 -*-
from mmvae_hub.base import BaseTrainer, BaseCallback


class PolymnistTrainer(BaseTrainer):
    def __init__(self, exp):
        super(PolymnistTrainer, self).__init__(exp)

    def _set_callback(self):
        return BaseCallback(self.exp)
