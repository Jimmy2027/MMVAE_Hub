# -*- coding: utf-8 -*-

import typing

import torch
from sklearn.metrics import average_precision_score

from mmvae_hub.utils import BaseMetrics


class MimicMetrics(BaseMetrics):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor, str_labels):
        super(MimicMetrics, self).__init__(prediction, groundtruth, str_labels)

    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics (accuracy, recall, specificity, precision, f1 score, jaccard score, dice score).
        NOTE: f1 and dice are the same
        """

        return {**{
            'accuracy': [self.accuracy()],
            'recall': [self.recall()],
            'specificity': [self.specificity()],
            'precision': [self.precision()],
            'f1': [self.f1()],
            'jaccard': [self.jaccard()],
            'dice': [self.dice()],

        },
                **self.mean_AP(), **self.get_counts()
                }

    def accuracy(self) -> float:
        """
        Computes the accuracy
        """
        self.INTER = torch.mul(self.prediction_bin, self.groundtruth_bin).sum()
        self.INTER_NEG = torch.mul(1 - self.prediction_bin, 1 - self.groundtruth_bin).sum()
        self.TOTAL = self.prediction_bin.nelement()
        return float(self.INTER + self.INTER_NEG) / float(self.TOTAL)

    def recall(self) -> float:
        """
        Computes the recall
        """
        self.TP = torch.mul(self.prediction_bin, self.groundtruth_bin).sum()
        self.FN = torch.mul(1 - self.prediction_bin, self.groundtruth_bin).sum()

        self.RC = float(self.TP) / (float(self.TP + self.FN) + 1e-6)

        return self.RC

    def specificity(self):
        self.TN = torch.mul(1 - self.prediction_bin, 1 - self.groundtruth_bin).sum()
        self.FP = torch.mul(self.prediction_bin, 1 - self.groundtruth_bin).sum()

        self.SP = float(self.TN) / (float(self.TN + self.FP) + 1e-6)

        return self.SP

    def precision(self) -> float:
        """
        Computes the precision
        """
        self.PC = float(self.TP) / (float(self.TP + self.FP) + 1e-6)

        return self.PC

    def f1(self) -> float:
        """
        Computes the f1 score (same as dice)
        """
        return 2 * (self.RC * self.PC) / (self.RC + self.PC + 1e-6)

    def jaccard(self) -> float:
        """
        Computes the jaccard score
        """
        return float(self.INTER) / (float(self.INTER + self.FP + self.FN) + 1e-6)

    def dice(self):
        """
        Computes the dice score (same as f1)
        """
        return 2 * float(self.INTER) / (float(2 * self.INTER + self.FP + self.FN) + 1e-6)

    def mean_AP(self) -> dict:
        """
        Compute the mean average precision.
        >>> import torch
        >>> metrics = Metrics(torch.tensor([0, 0, 1, 1]).unsqueeze(-1), torch.tensor([0.1, 0.4, 0.35, 0.8]).unsqueeze(-1), str_labels=['my_labels'])
        >>> metrics.mean_AP()
        {'mean_AP_my_labels': [0.8333333333333333], 'mean_AP_total': [0.8333333333333333]}
        """
        ap_values = {
            f'mean_AP_{self.str_labels[i]}': [
                average_precision_score(self.prediction[:, i].numpy().ravel(), self.groundtruth[:, i].numpy().ravel())]
            for i in range(len(self.str_labels))}
        ap_values['mean_AP_total'] = [average_precision_score(self.prediction.cpu().data.numpy().ravel(),
                                                              self.groundtruth.cpu().data.numpy().ravel())]
        return ap_values
