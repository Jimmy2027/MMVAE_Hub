import torch
import torch.nn as nn
from mmvae_hub.networks.utils.FeatureCompressor import LinearFeatureCompressor

from mmvae_hub.celeba.networks.DataGeneratorImg import DataGeneratorImg
from mmvae_hub.celeba.networks.FeatureExtractorImg import FeatureExtractorImg


class EncoderImg(nn.Module):
    def __init__(self, flags):
        super(EncoderImg, self).__init__()
        self.flags = flags
        self.feature_extractor = FeatureExtractorImg(flags, a=2.0, b=0.3)
        self.feature_compressor = LinearFeatureCompressor(flags.num_layers_img * flags.DIM_img,
                                                          0,
                                                          flags.class_dim)

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img)

        mu_content, logvar_content = self.feature_compressor(h_img)
        return None, None, mu_content, logvar_content


class DecoderImg(nn.Module):
    def __init__(self, flags):
        super(DecoderImg, self).__init__()
        self.flags = flags
        self.feature_generator = nn.Linear(flags.class_dim, flags.num_layers_img * flags.DIM_img,
                                           bias=True)
        self.img_generator = DataGeneratorImg(flags, a=2.0, b=0.3)

    def forward(self, z_style, z_content):
        z = z_content

        img_feat_hat = self.feature_generator(z)
        img_feat_hat = img_feat_hat.view(img_feat_hat.size(0), img_feat_hat.size(1), 1, 1)
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, torch.tensor(0.75).to(z.device)
