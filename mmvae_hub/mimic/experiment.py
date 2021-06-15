import random
import typing
from argparse import Namespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor

from mmvae_hub import log
from mmvae_hub.base.BaseExperiment import BaseExperiment
from mmvae_hub.mimic.MimicDataset import Mimic_testing, Mimic
from mmvae_hub.mimic.metrics import MimicMetrics
from mmvae_hub.mimic.modalities.MimicIMG import MimicPA, MimicLateral
from mmvae_hub.mimic.modalities.MimicText import MimicText
from mmvae_hub.mimic.utils import get_transform_img, get_str_labels
from mmvae_hub.modalities import BaseModality
from mmvae_hub.utils.utils import dict_to_device


class MimicExperiment(BaseExperiment):
    def __init__(self, flags):
        super().__init__(flags)
        self.labels = get_str_labels(flags.binary_labels)
        self.flags = flags
        self.dataset_name = 'mimic'
        self.plot_img_size = torch.Size((1, 128, 128))

        self.dataset_train, self.dataset_test = self.set_dataset()
        self.modalities: typing.Mapping[str, BaseModality] = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()

        self.mm_vae = self.set_model()

        self.clf_transforms: dict = self.set_clf_transforms()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = average_precision_score
        self.paths_fid = self.set_paths_fid()

        self.restart_experiment = False  # if true and the model returns nans, the workflow gets started again
        self.number_restarts = 0
        self.tb_logger = None

        self.metrics = MimicMetrics

    def set_modalities(self) -> typing.Mapping[str, BaseModality]:
        log.info('setting modalities')
        mod1 = MimicPA(self.flags, self.labels, self.flags.rec_weight_m1, self.plot_img_size)
        mod2 = MimicLateral(self.flags, self.labels, self.flags.rec_weight_m2, self.plot_img_size)
        mod3 = MimicText(self.flags, self.labels, self.flags.rec_weight_m3, self.plot_img_size,
                         self.dataset_train.report_findings_dataset.i2w)
        if self.flags.only_text_modality:
            mods = {mod3.name: mod3}
        else:
            mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}

        if self.flags.use_clf:
            for k, mod in mods.items():
                mod.set_clf()

        return mods

    def set_dataset(self):

        log.info('setting dataset')
        # used for faster unittests i.e. a dummy dataset
        if self.flags.dataset == 'toy':
            log.info('using testing dataset')
            self.flags.vocab_size = 3517
            d_train = Mimic_testing(self.flags)
            d_eval = Mimic_testing(self.flags)
        else:
            d_train = Mimic(self.flags, self.labels, split='train')
            d_eval = Mimic(self.flags, self.labels, split='eval')
        return d_train, d_eval

    def set_clf_transforms(self) -> dict:
        if self.flags.text_clf_type == 'word':
            def text_transform(x):
                # converts one hot encoding to indices vector
                return torch.argmax(x, dim=-1)
        else:
            def text_transform(x):
                return x

        # create temporary args to set the number of crops to 1
        temp_args = Namespace(**vars(self.flags))
        temp_args.n_crops = 1
        return {
            'PA': get_transform_img(temp_args, self.flags.img_clf_type),
            'Lateral': get_transform_img(temp_args, self.flags.img_clf_type),
            'text': text_transform
        }

    def set_rec_weights(self):
        """
        Sets the weights of the log probs for each modality.
        """
        log.info('setting rec_weights')

        return {
            'PA': self.flags.rec_weight_m1,
            'Lateral': self.flags.rec_weight_m2,
            'text': self.flags.rec_weight_m3
        }

    def set_style_weights(self):
        return {
            'PA': self.flags.beta_m1_style,
            'Lateral': self.flags.beta_m2_style,
            'text': self.flags.beta_m3_style,
        }

    def get_prediction_from_attr(self, values):
        return values.ravel()

    def get_test_samples(self, num_images=10) -> typing.Iterable[typing.Mapping[str, Tensor]]:
        """
        Gets random samples from the test dataset
        """
        n_test = self.dataset_test.__len__()
        samples = []
        for _ in range(num_images):
            sample, _ = self.dataset_test.__getitem__(random.randint(0, n_test - 1))
            sample = dict_to_device(sample, self.flags.device)

            samples.append(sample)

        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def eval_label(self, values: Tensor, labels: Tensor, index: int = None):
        """
        index: index of the labels
        """
        pred = values[:, index]
        gt = labels[:, index]
        return self.eval_metric(gt, pred)
