# -*- coding: utf-8 -*-
import typing

import torch
from mmvae_hub.utils import BaseMetrics
from sklearn.metrics import accuracy_score


class PolymnistMetrics(BaseMetrics):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    Modified version of https://github.com/ParGG/MasterThesisOld/blob/44f7b93214fa16494ebaeef7763ff81943b5ffc3/losses.py#L142
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor, str_labels):
        super(PolymnistMetrics, self).__init__(prediction, groundtruth, str_labels)

    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics.
        """

        return {**{
            'accuracy': [accuracy_score(self.groundtruth, self.prediction)],
        }, **self.get_counts()}
