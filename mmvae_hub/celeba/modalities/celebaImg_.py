import os
from pathlib import Path

import torch
# from mmvae_hub.mnistsvhntext.networks.ConvNetworksImgMNIST import EncoderImg
from mmvae_hub.celeba.networks.ConvNetworkImgClfCelebA import ClfImg
from mmvae_hub.modalities.ModalityIMG import ModalityIMG
from torchvision import transforms
from mmvae_hub.networks.utils.FeatureCompressor import LinearFeatureCompressor

from mmvae_hub.celeba.networks.ConvNetworksImgCelebA import DecoderImg
from mmvae_hub.utils.setup.flags_utils import get_config_path
from modun.file_io import json2dict

from modun.download_utils import download_zip_from_url, download_from_url

from mmvae_hub.polymnist.PolymnistMod import PolymnistMod
from PIL import Image
import torch
import torch.nn as nn

import torch.nn as nn

from mmvae_hub.celeba.networks.ResidualBlocks import ResidualBlock2dConv


def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0,
                                     b_val=0.3):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(
        ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample, a=a_val,
                            b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self, args, a, b):
        super(FeatureExtractorImg, self).__init__();
        self.args = args;
        self.a = a;
        self.b = b;
        self.conv1 = nn.Conv2d(self.args.image_channels, self.args.DIM_img,
                               kernel_size=3,
                               stride=2,
                               padding=2,
                               dilation=1,
                               bias=False)
        self.resblock1 = make_res_block_feature_extractor(args.DIM_img, 2 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock2 = make_res_block_feature_extractor(2 * args.DIM_img, 3 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock3 = make_res_block_feature_extractor(3 * args.DIM_img, 4 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock4 = make_res_block_feature_extractor(4 * args.DIM_img, 5 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=0, dilation=1, a_val=self.a, b_val=self.b)

    def forward(self, x):
        """
        torch.Size([256, 3, 64, 64])
        torch.Size([256, 128, 33, 33])
        torch.Size([256, 256, 16, 16])
        torch.Size([256, 384, 8, 8])
        torch.Size([256, 512, 4, 4])
        torch.Size([256, 640, 1, 1])
        """
        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out


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


class EncoderImg_(nn.Module):
    def __init__(self, flags):
        super(EncoderImg_, self).__init__()
        self.flags = flags
        self.feature_extractor = FeatureExtractorImg(flags, a=2.0, b=0.3)

        # self.hidden_dim = flags.num_layers_img * flags.DIM_img
        # modules = [nn.Sequential(nn.Linear(flags.image_channels * 64 * 64, self.hidden_dim), nn.ReLU(True))]
        # modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
        #                 for _ in range(flags.num_hidden_layers - 1)])
        # self.feature_extractor = nn.Sequential(*modules)

        # self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)
        # self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)
        self.feature_compressor = LinearFeatureCompressor(flags.num_layers_img * flags.DIM_img,
                                                          0,
                                                          flags.class_dim)

    def forward(self, x_img):
        # x_img = x_img.view(*x_img.size()[:-3], -1);
        h_img = self.feature_extractor(x_img)
        h = h_img.squeeze()
        mu_content, logvar_content = self.feature_compressor(h)
        return None, None, mu_content, logvar_content
        # latent_space_mu = self.hidden_mu(h);
        # latent_space_logvar = self.hidden_logvar(h);
        # latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
        # latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
        # return None, None, latent_space_mu, latent_space_logvar;


# class DecoderImg(nn.Module):
#     def __init__(self, flags):
#         super(DecoderImg, self).__init__();
#         self.flags = flags;
#         self.hidden_dim = 400;
#         modules = []
#         if flags.factorized_representation:
#             modules.append(
#                 nn.Sequential(nn.Linear(flags.style_mnist_dim + flags.class_dim, self.hidden_dim), nn.ReLU(True)))
#         else:
#             modules.append(nn.Sequential(nn.Linear(flags.class_dim, self.hidden_dim), nn.ReLU(True)))
#
#         modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
#                         for _ in range(flags.num_hidden_layers - 1)])
#         self.dec = nn.Sequential(*modules)
#         self.fc3 = nn.Linear(self.hidden_dim, 3 * 64 * 64)
#         self.relu = nn.ReLU();
#         self.sigmoid = nn.Sigmoid();
#
#     def forward(self, style_latent_space, class_latent_space):
#         z = class_latent_space;
#         x_hat = self.dec(z);
#         x_hat = self.fc3(x_hat);
#         x_hat = self.sigmoid(x_hat)
#         x_hat = x_hat.view(*z.size()[:-1], *torch.Size((3, 64, 64)))
#         return x_hat, torch.tensor(0.75).to(z.device);


class CelebaImg_(ModalityIMG):
    def __init__(self, flags, name):
        super().__init__(flags=flags, name=name, data_size=torch.Size((flags.image_channels, 64, 64)))
        self.plot_img_size = torch.Size((3, 64, 64))

        # rec weights will be set in CelebaExperiment
        self.rec_weight = None

        self.encoder = EncoderSVHN(flags).to(flags.device)
        self.decoder = DecoderImg(flags).to(flags.device)

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
