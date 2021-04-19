from abc import ABC, abstractmethod

import torch


class BaseModality(ABC):
    def __init__(self, flags, name):
        self.flags = flags
        self.name = name
        self.clf = self.set_clf()

    @abstractmethod
    def save_data(self, exp, d, fn, args):
        pass

    @abstractmethod
    def plot_data(self, exp, d):
        pass

    @abstractmethod
    def set_clf(self):
        pass

    @staticmethod
    def calc_log_prob(out_dist, target: torch.Tensor, norm_value: int):
        """
        Calculate log P(target | out_dist)
        """
        log_prob = out_dist.log_prob(target).sum()
        return log_prob / norm_value
