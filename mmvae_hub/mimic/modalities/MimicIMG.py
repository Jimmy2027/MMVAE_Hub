import os
import typing

import torch

from mmvae_hub.modalities.ModalityIMG import ModalityIMG
from mmvae_hub.modalities.utils import get_likelihood
from mmvae_hub.networks.images.CheXNet import CheXNet
from mmvae_hub.networks.images.ConvNetworkImgClf import ClfImg
from mmvae_hub.networks.images.ConvNetworksImgMimic import EncoderImg, DecoderImg
from mmvae_hub.utils.utils import get_clf_path


class MimicImg(ModalityIMG):
    def __init__(self, data_size, flags, name, labels, rec_weight, plot_img_size):
        super().__init__(data_size, flags, name)
        self.labels = labels
        self.likelihood_name = 'laplace'
        self.labels = labels
        self.gen_quality_eval = True
        self.file_suffix = '.png'

        self.encoder = EncoderImg(self.flags, self.flags.style_pa_dim).to(flags.device)
        self.decoder = DecoderImg(self.flags, self.flags.style_pa_dim).to(flags.device)

        self.likelihood = get_likelihood(self.likelihood_name)

        self.rec_weight = rec_weight

        self.plot_img_size = plot_img_size

        self.clf = self.get_clf()

    def get_clf(self):
        if self.flags.use_clf:
            # mapping clf type to clf_save_m*
            clf_save_names: typing.Mapping[str, str] = {
                'PA': self.flags.clf_save_m1,
                'Lateral': self.flags.clf_save_m2,
            }

            # finding the directory of the classifier
            dir_img_clf = os.path.join(self.flags.dir_clf,
                                       f'Mimic{self.flags.img_size}_{self.flags.img_clf_type}'
                                       f'{"_bin_label" if self.flags.binary_labels else ""}')
            dir_img_clf = os.path.expanduser(dir_img_clf)

            # finding and loading state dict
            clf = ClfImg(self.flags, self.labels) if self.flags.img_clf_type == 'resnet' else CheXNet(
                len(self.labels))
            clf_path = get_clf_path(dir_img_clf, clf_save_names[self.name])
            clf.load_state_dict(torch.load(clf_path, map_location=self.flags.device))
            return clf.to(self.flags.device)


class MimicPA(MimicImg):
    def __init__(self, flags, labels: typing.Iterable[str], rec_weight, plot_img_size):
        data_size = torch.Size((1, flags.img_size, flags.img_size))
        super().__init__(data_size=data_size, flags=flags, name='PA', labels=labels, rec_weight=rec_weight,
                         plot_img_size=plot_img_size)


class MimicLateral(MimicImg):
    def __init__(self, flags, labels: typing.Iterable[str], rec_weight, plot_img_size):
        data_size = torch.Size((1, flags.img_size, flags.img_size))
        super().__init__(data_size=data_size, flags=flags, name='Lateral', labels=labels, rec_weight=rec_weight,
                         plot_img_size=plot_img_size)
