import json
import random
from pathlib import Path

import PIL.Image as Image
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms

from mmvae_hub.base.BaseExperiment import BaseExperiment
from mmvae_hub.mnistsvhntext.MNISTmod import MNIST
# from mmvae_hub.celeba_.SVHNMNISTDataset import SVHNMNIST
from mmvae_hub.mnistsvhntext.SVHNMNISTDataset import SVHNMNIST
from mmvae_hub.mnistsvhntext.SVHNmod import SVHN
# from utils.BaseExperiment import BaseExperiment
from mmvae_hub.mnistsvhntext.metrics import mnistsvhntextMetrics
from mmvae_hub.mnistsvhntext.textmod import Text


class CelebA_(BaseExperiment):
    def __init__(self, flags):
        super().__init__(flags)
        self.flags = flags
        self.labels = ['digit']

        alphabet_path = Path(__file__).parent.parent / ('modalities/text/alphabet.json')
        with open(alphabet_path) as alphabet_file:
            self.alphabet = str(''.join(json.load(alphabet_file)))
        # self.flags.vocab_size = len(self.alphabet)

        self.dataset_train, self.dataset_test = self.set_dataset()

        self.plot_img_size = torch.Size((3, 28, 28))

        self.flags.num_features = len(self.alphabet)

        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()

        self.mm_vae = self.set_model()
        self.optimizer = None

        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score
        self.metrics = mnistsvhntextMetrics
        self.paths_fid = self.set_paths_fid()

    def set_modalities(self):
        mod1 = MNIST(self.flags, 'mnist')
        mod3 = Text(self.flags, self.alphabet)
        return {mod1.name: mod1, mod3.name: mod3}

    def get_transform_mnist(self):
        transform_mnist = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                                              transforms.ToTensor()])
        return transform_mnist

    def get_transform_svhn(self):
        transform_svhn = transforms.Compose([transforms.ToTensor()])
        return transform_svhn

    def set_dataset(self):
        transform_mnist = self.get_transform_mnist()
        transform_svhn = self.get_transform_svhn()
        transforms = [transform_mnist, transform_svhn]
        train = SVHNMNIST(self.flags,
                          self.alphabet,
                          train=True,
                          transform=transforms)
        test = SVHNMNIST(self.flags,
                         self.alphabet,
                         train=False,
                         transform=transforms)
        return train, test

    def set_rec_weights(self):
        rec_weights = dict()
        ref_mod_d_size = self.modalities['svhn'].data_size.numel()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        return rec_weights

    def set_style_weights(self):
        weights = dict()
        weights['mnist'] = None
        weights['svhn'] = None
        weights['text'] = None
        return weights

    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__()
        samples = []
        for i in range(num_images):

            sample, target = self.dataset_test.__getitem__(random.randint(0, n_test))

            samples.append(sample)

        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int)
        return pred

    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)
