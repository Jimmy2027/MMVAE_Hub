import os
import random
from pathlib import Path

import mmvae_base
import numpy as np
import torch
import torch.optim as optim
from PIL import ImageFont
from mmvae_base import BaseExperiment
from sklearn.metrics import accuracy_score
from torchvision import transforms

from mmvae_mst.mmnist.MMNISTDataset import MMNISTDataset, ToyMMNISTDataset
from mmvae_mst.mmnist.metrics import MmnistMetrics
from mmvae_mst.mmnist.networks.ConvNetworkImgClfCMNIST import ClfImg as ClfImgCMNIST
from mmvae_mst.mmnist.networks.ConvNetworksImgCMNIST import EncoderImg, DecoderImg
from mmvae_mst.mmnist.networks.VAEMMNIST import VAEMMNIST
from mmvae_mst.modalities.CMNIST import CMNIST


class MMNISTExperiment(BaseExperiment):
    def __init__(self, flags, alphabet):
        super(MMNISTExperiment, self).__init__(flags)
        self.flags = flags
        self.labels = ['digit']
        # self.name = flags.name
        self.dataset_name = 'mmnist'
        self.num_modalities = flags.num_mods
        self.plot_img_size = torch.Size((3, 28, 28))
        font_path = str(Path(mmvae_base.__file__).parent / 'modalities/text/FreeSerif.ttf')
        self.font = ImageFont.truetype(font_path, 38)
        self.alphabet = alphabet
        self.flags.num_features = len(alphabet)

        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()

        self.dataset_train, self.dataset_test = self.set_dataset()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score;
        self.metrics = MmnistMetrics
        self.paths_fid = self.set_paths_fid()

    def set_model(self):
        model = VAEMMNIST(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model

    def set_modalities(self):
        mods = [CMNIST(EncoderImg(self.flags), DecoderImg(self.flags), name="m%d" % m) for m in
                range(self.num_modalities)]
        return {m.name: m for m in mods}

    def set_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.flags.dataset == 'toy':
            train = ToyMMNISTDataset()
            test = ToyMMNISTDataset()
        else:
            train = MMNISTDataset(Path(self.flags.dir_data) / 'train', transform=transform)
            test = MMNISTDataset(Path(self.flags.dir_data) / 'train', transform=transform)
        return train, test

    def set_clfs(self):
        clfs = {"m%d" % m: None for m in range(self.num_modalities)}
        if self.flags.use_clf:
            for m in range(self.num_modalities):
                model_clf = ClfImgCMNIST()
                model_clf.load_state_dict(
                    torch.load(os.path.join(self.flags.dir_clf, "pretrained_img_to_digit_clf_m%d" % m)))
                model_clf = model_clf.to(self.flags.device)
                clfs["m%d" % m] = model_clf
        return clfs

    def set_optimizer(self):
        # optimizer definition
        params = []
        for model in [*self.mm_vae.encoders, *self.mm_vae.decoders]:
            for p in model.parameters():
                params.append(p)
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

    def get_transform_mmnist(self):
        return transforms.Compose([transforms.ToTensor()])

    def get_test_samples(self, num_images=10):
        n_test = len(self.dataset_test)
        samples = []
        for i in range(num_images):
            while True:
                ix = random.randint(0, n_test - 1)
                sample, target = self.dataset_test[ix]
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key].to(self.flags.device)
                    samples.append(sample)
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

    def eval_label(self, values, labels, index):
        # todo: HK: abstract method that needs to be implemented. No idea for what it is needed
        pass
