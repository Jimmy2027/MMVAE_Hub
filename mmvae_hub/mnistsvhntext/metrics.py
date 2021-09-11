# -*- coding: utf-8 -*-
import typing

import torch
from mmvae_hub.utils import BaseMetrics
from sklearn.metrics import accuracy_score


class mnistsvhntextMetrics(BaseMetrics):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor, str_labels):
        super(mnistsvhntextMetrics, self).__init__(prediction, groundtruth, str_labels)

    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics.
        """

        return {**{
            'accuracy': [accuracy_score(self.groundtruth, self.prediction)],
        }, **self.get_counts()}
