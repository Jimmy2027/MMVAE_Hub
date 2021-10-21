# -*- coding: utf-8 -*-
import typing

import torch
from sklearn.metrics import average_precision_score

from mmvae_hub.utils import BaseMetrics


class CelebAMetrics(BaseMetrics):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor, str_labels):
        super(CelebAMetrics, self).__init__(prediction, groundtruth, str_labels)

    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics.
        """

        return {**{
            'avg_prec': [average_precision_score(self.groundtruth, self.prediction)],
        }, **self.get_counts()}
