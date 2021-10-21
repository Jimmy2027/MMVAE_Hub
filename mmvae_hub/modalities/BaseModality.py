from abc import ABC, abstractmethod
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor


class BaseModality(ABC):
    def __init__(self, flags, name):
        self.data_size = None
        self.flags = flags
        self.name = name
        self.rec_weight = None

        self.encoder = None
        self.decoder = None

        self.px_z = None

    @abstractmethod
    def save_data(self, d, fn, args):
        pass

    @abstractmethod
    def plot_data(self, exp, d):
        pass

    @abstractmethod
    def get_clf(self):
        pass

    @staticmethod
    def calc_log_prob(out_dist, target: torch.Tensor, norm_value: int):
        """
        Calculate log P(target | out_dist)
        """
        log_prob = out_dist.log_prob(target).sum()
        return log_prob / norm_value

    def calc_likelihood(self, class_embeddings, unflatten: Tuple = None):
        if unflatten:
            mu, scale = self.decoder(class_embeddings)
            return self.px_z(loc=mu.unflatten(0, unflatten), scale=scale)
        return self.px_z(*self.decoder(class_embeddings), validate_args=False)

    def log_likelihood(self, px_z, batch_sample):
        return px_z.log_prob(batch_sample)

    def plot_data_single_img(self, d: Tensor):
        return plt.imshow(self.plot_data(d.squeeze(dim=0)).cpu().detach().squeeze())
