import os

import torch

from mmvae_hub.base.modalities import BaseModality
from mmvae_hub.polymnist.networks.ConvNetworkImgClfPolymnist import ClfImg
from mmvae_hub.polymnist.networks.ConvNetworksImgPolymnist import EncoderImg, DecoderImg
from mmvae_hub.polymnist.utils import download_polymnist_clfs
from mmvae_hub.utils import utils
from mmvae_hub.utils.save_samples import write_samples_img_to_file


class PolyMNISTMod(BaseModality):
    def __init__(self, flags, name: str):
        super(PolyMNISTMod, self).__init__(flags, name)

        self.likelihood_name = 'laplace'
        self.data_size = torch.Size((3, 28, 28))
        self.gen_quality_eval = True
        self.file_suffix = '.png'
        self.encoder = EncoderImg(flags)
        self.decoder = DecoderImg(flags)
        self.likelihood = utils.get_likelihood(self.likelihood_name)
        self.clf = self.set_clf()
        # self.transform = transforms.Compose([transforms.ToTensor()])

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d):
        # out = self.transform(d.squeeze(0).cpu()).cuda().unsqueeze(0)
        # return out
        return d

    def set_clf(self):
        dir_clf = self.flags.dir_clf
        if not dir_clf.exists():
            download_polymnist_clfs(dir_clf)
        model_clf = ClfImg()
        model_clf.load_state_dict(
            torch.load(os.path.join(self.flags.dir_clf, f"pretrained_img_to_digit_clf_{self.name}"),
                       map_location=self.flags.device))
        return model_clf.to(self.flags.device)
