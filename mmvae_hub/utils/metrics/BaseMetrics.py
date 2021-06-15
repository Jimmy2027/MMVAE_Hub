# -*- coding: utf-8 -*-
import typing
from abc import abstractmethod
from typing import List

import torch


class BaseMetrics(object):
    """
    Defines a set of metrics that are used to evaluate the performance of a model
    """

    def __init__(self, prediction: torch.Tensor, groundtruth: torch.Tensor, str_labels: List[str]):
        """
        params:
            prediction: Tensor which is given as output of the network
            groundtruth: Tensor which resembles the goundtruth
        >>> import torch
        >>> metrics = Metrics(torch.ones((10,1)), torch.ones((10,1)), str_labels=['my_labels'])
        """
        self.str_labels = str_labels
        self.prediction = prediction
        self.groundtruth = groundtruth
        self.prediction_bin: torch.Tensor = (prediction > 0.5) * 1
        self.groundtruth_bin: torch.Tensor = (groundtruth > 0.5) * 1

        # classwise binarized predictions
        self.class_pred_bin: dict = {str_labels[i]: self.prediction_bin[:, i] for i in range(len(str_labels))}
        self.class_gt_bin: dict = {str_labels[i]: self.groundtruth_bin[:, i] for i in range(len(str_labels))}

    @abstractmethod
    def evaluate(self) -> typing.Dict[str, list]:
        """
        Computes the different metrics.
        """
        pass

    def get_counts(self) -> dict:
        predicted_counts = {f'pred_count_{label}': [self.class_pred_bin[label].sum().item()] for label in
                            self.str_labels}
        gt_counts = {f'gt_count_{label}': [self.class_gt_bin[label].sum().item()] for label in self.str_labels}

        return {**predicted_counts, **gt_counts}

    def extract_values(self, results: dict):
        """
        Extract first values from list for each metric result.
        >>> import torch
        >>> metrics = Metrics(torch.ones((10,1)), torch.ones((10,1)), str_labels=['my_labels'])
        >>> metrics.extract_values(results={'accuracy':[0.9], 'f1': [0.8], 'recall':[0.6]})
        {'accuracy': 0.9, 'f1': 0.8, 'recall': 0.6}
        """
        return {k: v[0] for k, v in results.items()}
