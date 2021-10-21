import os
from pathlib import Path

import torch
from modun.download_utils import download_zip_from_url
from modun.file_io import json2dict

from mmvae_hub.mnistsvhntext.networks.ConvNetworkImgClfMNIST import ClfImg
from mmvae_hub.mnistsvhntext.networks.ConvNetworksImgMNIST import DecoderImg, EncoderImg
from mmvae_hub.polymnist.PolymnistMod import PolymnistMod
from mmvae_hub.utils.setup.flags_utils import get_config_path


class MNIST(PolymnistMod):
    def __init__(self, flags, name):
        super().__init__(flags=flags, name=name)
        self.rec_weight = 1.
        self.encoder = EncoderImg(flags).to(flags.device)
        self.decoder = DecoderImg(flags).to(flags.device)

        self.data_size = torch.Size((1, 28, 28))
        self.gen_quality_eval = True
        self.file_suffix = '.png'

    def plot_data(self, d):
        return d.repeat(1, 3, 1, 1)

    def get_clf(self):
        if self.flags.use_clf:
            dir_clf = self.flags.dir_clf
            if not dir_clf.exists():
                download_zip_from_url(
                    url='https://www.dropbox.com/sh/lx8669lyok9ois6/AADM7Cs_QReijyo2kF8xzWqua/trained_classifiers/trained_clfs_mst?dl=1',
                    dest_folder=dir_clf)
            model_clf = ClfImg()
            model_clf.load_state_dict(
                torch.load(os.path.join(self.flags.dir_clf, f"clf_m1"),
                           map_location=self.flags.device))

            return model_clf.to(self.flags.device)


if __name__ == '__main__':
    config = json2dict(Path(get_config_path(dataset='mnistsvhntext')))
    download_zip_from_url(
        url='https://www.dropbox.com/sh/lx8669lyok9ois6/AADM7Cs_QReijyo2kF8xzWqua/trained_classifiers/trained_clfs_mst?dl=1',
        dest_folder=Path(config['dir_clf']).expanduser())
