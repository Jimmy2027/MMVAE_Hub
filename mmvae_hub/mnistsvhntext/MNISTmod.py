import os
from pathlib import Path

import torch
from mmvae_hub.networks.utils.ResidualBlocks import ResidualBlock2dConv

from mmvae_hub.utils.setup.flags_utils import get_config_path
from modun.file_io import json2dict

from mmvae_hub.mnistsvhntext.networks.ConvNetworkImgClfMNIST import ClfImg
from modun.download_utils import download_zip_from_url

from mmvae_hub.mnistsvhntext.networks.ConvNetworksImgMNIST import DecoderImg

from mmvae_hub.polymnist.PolymnistMod import PolymnistMod
import os
from pathlib import Path

import torch
# from mmvae_hub.mnistsvhntext.networks.ConvNetworksImgMNIST import EncoderImg

from mmvae_hub.networks.utils.FeatureCompressor import LinearFeatureCompressor

# from mmvae_hub.celeba.networks.ConvNetworksImgCelebA import DecoderImg
from mmvae_hub.utils.setup.flags_utils import get_config_path
from modun.file_io import json2dict

from mmvae_hub.mnistsvhntext.networks.ConvNetworkImgClfMNIST import ClfImg
from modun.download_utils import download_zip_from_url

from mmvae_hub.polymnist.PolymnistMod import PolymnistMod

import torch
import torch.nn as nn

import torch.nn as nn



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
        self.conv1 = nn.Conv2d(1, self.args.DIM_img,
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
        # self.resblock4 = make_res_block_feature_extractor(4 * args.DIM_img, 5 * args.DIM_img, kernelsize=4, stride=2,
        #                                                   padding=0, dilation=1, a_val=self.a, b_val=self.b)

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
        # out = self.resblock4(out);
        return out

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


class MNIST(PolymnistMod):
    def __init__(self, flags, name):
        super().__init__(flags=flags, name=name)
        self.rec_weight = 1.
        # self.encoder = EncoderImg(flags).to(flags.device)
        flags.DIM_img = 28
        flags.num_layers_img = 4
        self.encoder = EncoderImg_(flags).to(flags.device)
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
