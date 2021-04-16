# -*- coding: utf-8 -*-
from mmvae_mst.networks.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from mmvae_mst.networks.ConvNetworkImgClfMNIST import ClfImg as ClfImgMNIST
from mmvae_mst.networks.ConvNetworkImgClfSVHN import ClfImgSVHN
from mmvae_mst.networks.ConvNetworkTextClf import ClfText as ClfText

from mmvae_mst.networks.ConvNetworksImgMNIST import EncoderImg, DecoderImg
from mmvae_mst.networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
from mmvae_mst.networks.ConvNetworksTextMNIST import EncoderText, DecoderText
