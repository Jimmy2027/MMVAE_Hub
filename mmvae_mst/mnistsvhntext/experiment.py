import os
import random
from pathlib import Path

import PIL.Image as Image
import mmvae_base
import numpy as np
import torch
import torch.optim as optim
from PIL import ImageFont
from mmvae_base import BaseExperiment
from mmvae_mst.mnistsvhntext.SVHNMNISTDataset import SVHNMNIST
from mmvae_mst.modalities.MNIST import MNIST
from mmvae_mst.modalities.SVHN import SVHN
from mmvae_mst.modalities.Text import Text
from mmvae_mst.mnistsvhntext.networks import VAEtrimodalSVHNMNIST, ClfImgMNIST, ClfImgSVHN, ClfText, EncoderImg, DecoderImg, \
    EncoderSVHN, DecoderSVHN, EncoderText, DecoderText
from sklearn.metrics import accuracy_score
from torchvision import transforms


# from utils.BaseExperiment import BaseExperiment


class MNISTSVHNText(BaseExperiment):
    def __init__(self, flags, alphabet):
        super().__init__(flags)
        self.plot_img_size = torch.Size((3, 28, 28))
        self.font = self.get_font()
        self.alphabet = alphabet;
        self.flags.num_features = len(alphabet);

        self.modalities = self.set_modalities();
        self.num_modalities = len(self.modalities.keys());
        self.subsets = self.set_subsets();
        self.dataset_train = None;
        self.dataset_test = None;
        self.set_dataset();

        self.mm_vae = self.set_model();
        self.clfs = self.set_clfs();
        self.optimizer = None;
        self.rec_weights = self.set_rec_weights();
        self.style_weights = self.set_style_weights();

        self.test_samples = self.get_test_samples();
        self.eval_metric = accuracy_score;
        self.paths_fid = self.set_paths_fid();

        self.labels = ['digit'];

    def set_model(self):
        model = VAEtrimodalSVHNMNIST(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device);
        return model;

    def set_modalities(self):
        mod1 = MNIST(EncoderImg(self.flags), DecoderImg(self.flags));
        mod2 = SVHN(EncoderSVHN(self.flags), DecoderSVHN(self.flags),
                    self.plot_img_size);
        mod3 = Text(EncoderText(self.flags), DecoderText(self.flags),
                    self.flags.len_sequence,
                    self.alphabet,
                    self.plot_img_size,
                    self.font);
        mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3};
        return mods;

    def get_transform_mnist(self):
        transform_mnist = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                                              transforms.ToTensor()])
        return transform_mnist;

    def get_transform_svhn(self):
        transform_svhn = transforms.Compose([transforms.ToTensor()])
        return transform_svhn;

    def set_dataset(self):
        transform_mnist = self.get_transform_mnist();
        transform_svhn = self.get_transform_svhn();
        transforms = [transform_mnist, transform_svhn];
        train = SVHNMNIST(self.flags,
                          self.alphabet,
                          train=True,
                          transform=transforms)
        test = SVHNMNIST(self.flags,
                         self.alphabet,
                         train=False,
                         transform=transforms)
        self.dataset_train = train;
        self.dataset_test = test;

    def set_clfs(self):
        model_clf_m1 = None;
        model_clf_m2 = None;
        model_clf_m3 = None;
        if self.flags.use_clf:
            model_clf_m1 = ClfImgMNIST();
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m1)))
            model_clf_m1 = model_clf_m1.to(self.flags.device);

            model_clf_m2 = ClfImgSVHN();
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m2)))
            model_clf_m2 = model_clf_m2.to(self.flags.device);

            model_clf_m3 = ClfText(self.flags);
            model_clf_m3.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m3)))
            model_clf_m3 = model_clf_m3.to(self.flags.device);

        clfs = {'mnist': model_clf_m1,
                'svhn': model_clf_m2,
                'text': model_clf_m3}
        return clfs;

    def set_optimizer(self):
        # optimizer definition
        optimizer = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        self.optimizer = optimizer;

    def set_rec_weights(self):
        rec_weights = dict();
        ref_mod_d_size = self.modalities['svhn'].data_size.numel();
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        return rec_weights;

    def set_style_weights(self):
        weights = dict();
        weights['mnist'] = self.flags.beta_m1_style;
        weights['svhn'] = self.flags.beta_m2_style;
        weights['text'] = self.flags.beta_m3_style;
        return weights;

    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__();
        samples = []
        for i in range(num_images):
            while True:
                sample, target = self.dataset_test.__getitem__(random.randint(0, n_test))
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key].to(self.flags.device);
                    samples.append(sample)
                    break;
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values));

    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int);
        return pred;

    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values);
        return self.eval_metric(labels, pred);

    def get_font(self):
        font_path = Path(mmvae_base.__file__).parent / 'modalities/text/FreeSerif.ttf'
        return ImageFont.truetype(str(font_path), 38)
