# -*- coding: utf-8 -*-
import os

import torch


class BaseCallback:
    def __init__(self, exp):
        self.exp = exp
        self.flags = exp.flags

    def update_epoch(self, epoch):
        # save checkpoints after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == self.flags.end_epoch:
            dir_network_epoch = os.path.join(self.flags.dir_checkpoints, str(epoch).zfill(4))
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch)
            self.exp.mm_vae.save_networks()
            torch.save(self.exp.mm_vae.state_dict(), os.path.join(dir_network_epoch, self.flags.mm_vae_save))
