import os
import random
from pathlib import Path
from typing import Mapping, Iterable

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch import Tensor
from torchvision import transforms

from mmvae_hub.base.BaseExperiment import BaseExperiment
from mmvae_hub.base.modalities.BaseModality import BaseModality
from mmvae_hub.base.utils.utils import dict_to_device
from mmvae_hub.polymnist.PolymnistDataset import PolymnistDataset, ToyPolymnistDataset
from mmvae_hub.polymnist.PolymnistMod import PolymnistMod
from mmvae_hub.polymnist.metrics import PolymnistMetrics


class PolymnistExperiment(BaseExperiment):
    def __init__(self, flags):
        super(PolymnistExperiment, self).__init__(flags)
        self.flags = flags
        self.labels = ['digit']
        # self.name = flags.name
        self.dataset_name = 'polymnist'
        self.num_modalities = flags.num_mods
        self.plot_img_size = torch.Size((3, 28, 28))

        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()

        self.dataset_train, self.dataset_test = self.set_dataset()

        self.mm_vae = self.set_model()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score
        self.metrics = PolymnistMetrics
        self.paths_fid = self.set_paths_fid()

    def set_modalities(self) -> Mapping[str, BaseModality]:
        mods = [PolymnistMod(self.flags, name="m%d" % m) for m in range(self.num_modalities)]
        return {m.name: m for m in mods}

    def set_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.flags.dataset == 'toy':
            train = ToyPolymnistDataset(num_modalities=self.num_modalities)
            test = ToyPolymnistDataset(num_modalities=self.num_modalities)
        else:
            train = PolymnistDataset(Path(self.flags.dir_data) / 'train', transform=transform,
                                     num_modalities=self.num_modalities)
            test = PolymnistDataset(Path(self.flags.dir_data) / 'train', transform=transform,
                                    num_modalities=self.num_modalities)
        return train, test

    def set_optimizer(self):
        # optimizer definition
        params = []
        for _, mod in self.modalities.items():
            for model in [mod.encoder, mod.decoder]:
                for p in model.parameters():
                    params.append(p)

        # add flow parameters from mmvae if present
        params.extend(list(self.mm_vae.parameters()))

        optimizer = optim.Adam(params, lr=self.flags.initial_learning_rate, betas=(self.flags.beta_1,
                                                                                   self.flags.beta_2))
        self.optimizer = optimizer

    def set_rec_weights(self):
        rec_weights = {}
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = 1.0
        return rec_weights

    def set_style_weights(self):
        return {"m%d" % m: self.flags.beta_style for m in range(self.num_modalities)}

    def get_transform_polymnist(self):
        return transforms.Compose([transforms.ToTensor()])

    def get_test_samples(self, num_images=10) -> Iterable[Mapping[str, Tensor]]:
        n_test = len(self.dataset_test)
        samples = []
        for i in range(num_images):
            while True:
                # loop until sample with label i is found
                ix = random.randint(0, n_test - 1)
                sample, target = self.dataset_test[ix]
                if target == i:
                    samples.append(dict_to_device(sample, self.flags.device))
                    break
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def set_paths_fid(self):
        dir_real = os.path.join(self.flags.dir_gen_eval_fid, 'real')
        dir_random = os.path.join(self.flags.dir_gen_eval_fid, 'random')
        paths = {'real': dir_real,
                 'random': dir_random}
        dir_cond = self.flags.dir_gen_eval_fid
        for k, name in enumerate(self.subsets):
            paths[name] = os.path.join(dir_cond, name)
        print(paths.keys())
        return paths

    def get_prediction_from_attr(self, attr, index=None):
        return np.argmax(attr, axis=1).astype(int);

    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values);
        return self.eval_metric(labels, pred);
