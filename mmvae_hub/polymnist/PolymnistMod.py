import os

import torch
from torch import distributions as dist

from mmvae_hub.modalities.ModalityIMG import ModalityIMG
from mmvae_hub.modalities.utils import get_likelihood
from mmvae_hub.polymnist.networks.ConvNetworkImgClfPolymnist import ClfImg
from mmvae_hub.polymnist.networks.ConvNetworksImgPolymnist import EncoderImg, DecoderImg
from mmvae_hub.polymnist.utils import download_polymnist_clfs
from mmvae_hub.utils.plotting.save_samples import write_samples_img_to_file


class PolymnistMod(ModalityIMG):
    def __init__(self, flags, name: str):
        super(PolymnistMod, self).__init__(data_size=torch.Size((3, 28, 28)), flags=flags, name=name)

        self.likelihood_name = 'laplace'
        self.gen_quality_eval = True
        self.file_suffix = '.png'
        self.encoder = EncoderImg(flags).to(flags.device)
        self.decoder = DecoderImg(flags).to(flags.device)

        self.pz = dist.Laplace  # prior
        self.px_z = dist.Laplace  # likelihood
        self.qz_x = dist.Laplace  # posterior

        self.likelihood = get_likelihood(self.likelihood_name)
        self.rec_weight = 1.0
        # self.transform = transforms.Compose([transforms.ToTensor()])

        self.clf = self.get_clf()

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d):
        # out = self.transform(d.squeeze(0).cpu()).cuda().unsqueeze(0)
        # return out
        return d

    def get_clf(self):
        if self.flags.use_clf:
            dir_clf = self.flags.dir_clf
            if not dir_clf.exists():
                download_polymnist_clfs(dir_clf)
            model_clf = ClfImg()
            model_clf.load_state_dict(
                torch.load(os.path.join(self.flags.dir_clf, f"pretrained_img_to_digit_clf_{self.name}"),
                           map_location=self.flags.device))

            return model_clf.to(self.flags.device)
