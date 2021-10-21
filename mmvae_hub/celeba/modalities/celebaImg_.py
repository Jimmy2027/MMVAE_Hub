from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from modun.download_utils import download_zip_from_url, download_from_url
from modun.file_io import json2dict
from torchvision import transforms

# from mmvae_hub.mnistsvhntext.networks.ConvNetworksImgMNIST import EncoderImg
from mmvae_hub.celeba.networks.ConvNetworkImgClfCelebA import ClfImg
from mmvae_hub.celeba.networks.ConvNetworksImgCelebA import DecoderImg
from mmvae_hub.mnisttext_.SVHNmod import DecoderSVHN
from mmvae_hub.modalities.ModalityIMG import ModalityIMG
from mmvae_hub.utils.setup.flags_utils import get_config_path


class EncoderSVHN(nn.Module):
    def __init__(self, flags):
        super(EncoderSVHN, self).__init__()
        self.flags = flags;
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
        self.conv5 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0, dilation=1);
        self.relu = nn.ReLU();

        # non-factorized
        self.hidden_mu = nn.Linear(in_features=256, out_features=flags.class_dim, bias=True)
        self.hidden_logvar = nn.Linear(in_features=256, out_features=flags.class_dim, bias=True)

    def forward(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.relu(h);
        h = self.conv5(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        latent_space_mu = self.hidden_mu(h);
        latent_space_logvar = self.hidden_logvar(h);
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
        return None, None, latent_space_mu, latent_space_logvar;


class CelebaImg_(ModalityIMG):
    def __init__(self, flags, name):
        super().__init__(flags=flags, name=name, data_size=torch.Size((flags.image_channels, 64, 64)))
        self.plot_img_size = torch.Size((3, 64, 64))

        # rec weights will be set in CelebaExperiment
        self.rec_weight = None

        self.encoder = EncoderSVHN(flags).to(flags.device)
        # self.encoder = EncoderImg_(flags).to(flags.device)
        self.decoder = DecoderSVHN(flags).to(flags.device)
        # self.decoder = DecoderImg(flags).to(flags.device)

        self.transform_plot = self.get_plot_transform()
        self.data_size = torch.Size((flags.image_channels, 64, 64))
        # self.data_size = torch.Size((flags.image_channels, 28,28))
        self.gen_quality_eval = True
        self.file_suffix = '.png'
        self.clf = self.get_clf()

    def plot_data(self, d):
        return self.transform_plot(d.squeeze(0).cpu()).cuda().unsqueeze(0)

    def get_plot_transform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size=list(self.plot_img_size)[1:], interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )

    def get_clf(self):
        if self.flags.use_clf:
            img_clf_path = self.flags.dir_clf / 'clf_celeba_img.pth'
            if not img_clf_path.exists():
                download_from_url(
                    url='https://www.dropbox.com/sh/lx8669lyok9ois6/AABxwQKXU1no5cM91eStMhfIa/trained_classifiers/trained_clfs_celeba/clf_m1?dl=1',
                    dest_path=img_clf_path, verbose=True)

            clf = ClfImg(self.flags)
            clf.load_state_dict(torch.load(img_clf_path, map_location=self.flags.device))

            return clf.to(self.flags.device)


if __name__ == '__main__':
    config = json2dict(Path(get_config_path(dataset='mnistsvhntext')))
    download_zip_from_url(
        url='https://www.dropbox.com/sh/lx8669lyok9ois6/AADM7Cs_QReijyo2kF8xzWqua/trained_classifiers/trained_clfs_mst?dl=1',
        dest_folder=Path(config['dir_clf']).expanduser())
