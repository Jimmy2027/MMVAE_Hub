import random
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from mmvae_hub.VQVAE.VqVaeDataclasses import VQEncMod, JointEmbeddingVQ, JointLatentsVQ, QuantizedLatent, \
    VQForwardResults
from mmvae_hub.mimic.MimicDataset import Mimic
from mmvae_hub.mimic.utils import get_str_labels
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.flows.ConvFlow import ConvFlow
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods


class VQVAE(BaseMMVAE):
    """Multi modal VQ-VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

        self.vector_quantizer = VectorQuantizerEMA(flags.num_embeddings, flags.embedding_dim,
                                                   flags.commitment_cost, flags.decay)

    def forward(self, input_batch: dict) -> VQForwardResults:
        enc_mods, joint_latents = self.inference(input_batch)

        # quantize joint_latents
        quantized_latents = self.quantize(joint_latents)

        # reconstruct modalities
        rec_mods = self.decode(batch_mods=[mod_str for mod_str in input_batch], quantized_latents=quantized_latents)

        return VQForwardResults(enc_mods=enc_mods, quantized_latents=quantized_latents, rec_mods=rec_mods)

    def quantize(self, joint_latents: JointLatentsVQ) -> Mapping[str, QuantizedLatent]:
        quantized_latents = {}
        for s_key, subset in joint_latents.subsets.items():
            quantizer_loss, quantized, perplexity, _ = self.vector_quantizer(subset)
            quantized_latents[s_key] = QuantizedLatent(quantizer_loss=quantizer_loss, quantized=quantized,
                                                       perplexity=perplexity)

        return quantized_latents

    def encode(self, input_batch: Mapping[str, Tensor]) -> Mapping[str, VQEncMod]:
        enc_mods = {}
        for mod_str, mod in self.modalities.items():
            if mod_str in input_batch:
                enc_mods[mod_str] = {}

                z = mod.encoder(input_batch[mod_str])

                enc_mods[mod_str] = VQEncMod(enc_mod=z)

        return enc_mods

    def decode(self, batch_mods: typing.Iterable[str], quantized_latents: Mapping[str, QuantizedLatent]) \
            -> Mapping[str, Mapping[str, Tensor]]:
        """The decoder decodes every modality from every subset."""
        rec_mods = {s_key: {} for s_key in quantized_latents}

        for s_key, q_subset in quantized_latents.items():
            for mod_str in batch_mods:
                mod = self.modalities[mod_str]
                rec_mods[s_key][mod_str] = mod.decoder(quantized_latents[s_key].quantized)
        return rec_mods

    def calculate_loss(self, forward_results: VQForwardResults, batch_d: dict) \
            -> tuple[float, dict, dict]:
        """Average all quantizer losses and all reconstruction errors for all subsets to get the total loss."""
        quant_losses = {}
        rec_losses = {s_key: {} for s_key in forward_results.quantized_latents}
        # total_rec_loss = torch.Tensor().to(self.flags.device)
        total_quant_loss = torch.Tensor().to(self.flags.device)
        # temp try to optimize only the reconstruction of missing modalities
        total_rec_loss = torch.Tensor().to(self.flags.device)

        for s_key, quantized_latent in forward_results.quantized_latents.items():
            quant_losses[s_key] = quantized_latent.quantizer_loss
            total_quant_loss = torch.cat((quantized_latent.quantizer_loss.unsqueeze(0), total_quant_loss))

            for mod_str, rec_mod in forward_results.rec_mods[s_key].items():
                if mod_str == 'text':
                    rec_mod = rec_mod.argmax(dim=1)
                rec_loss = F.mse_loss(rec_mod, batch_d[mod_str])
                # total_rec_loss = torch.cat((rec_loss.unsqueeze(0), total_rec_loss))
                rec_losses[s_key][mod_str] = rec_loss

                # temp try to optimize only the reconstruction of missing modalities
                if mod_str not in s_key.split('_'):
                    total_rec_loss = torch.cat((rec_loss.unsqueeze(0), total_rec_loss))

        # take a random subset to use its average reconstruction loss for the gradients.
        # selected_subset_rec_loss = torch.cat(
        #     [v.unsqueeze(0) for _, v in rec_losses[random.choice(list(rec_losses.keys()))].items()])
        # try to optimize only the reconstruction of missing modalities
        total_loss = total_rec_loss.mean(0) + total_quant_loss.mean(0)

        return total_loss, quant_losses, rec_losses

    # ==================================================================================================================
    # Generation
    # ===============================================================================================================
    def conditioned_generation(self, input_samples: dict, subset_key: str, style=None):
        """
        Generate samples conditioned with input samples for a given subset.

        subset_key str: The key indicating which subset is used for the generation.
        """

        enc_mods, joint_latents = self.inference(input_samples)
        joint_latents.subsets = {k: v for k, v in joint_latents.subsets.items() if k == subset_key}

        # quantize joint_latents
        quantized_latents = self.quantize(joint_latents)

        # reconstruct modalities
        cond_gen = {}
        for mod_str, mod in self.modalities.items():
            cond_gen[mod_str] = mod.decoder(quantized_latents[subset_key].quantized)

        return cond_gen

    def cond_generation(self, joint_latent: JointLatentsVQ, num_samples=None) -> Mapping[str, Mapping[str, Tensor]]:
        """Generate from joint_latent"""
        if num_samples is None:
            num_samples = self.flags.batch_size

        # quantize joint_latents
        quantized_latents = self.quantize(joint_latent)

        # reconstruct modalities
        cond_gen_samples = self.decode(batch_mods=[mod_str for mod_str in self.modalities],
                                       quantized_latents=quantized_latents)

        return cond_gen_samples

    def generate(self, num_samples=None):
        """Not yet implemented for VQ-VAE."""
        if num_samples is None:
            num_samples = self.flags.batch_size

        cond_gen = {}
        for mod_str, mod in self.modalities.items():
            cond_gen[mod_str] = torch.ones((num_samples, 1, self.flags.img_size))
        return


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VQMoGFMVAE(VQVAE):
    """Vector Quantized Mixture of Generalized f-Means VAE"""

    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)
        self.flow = ConvFlow(class_dim=64, num_flows=flags.num_gfm_flows, coupling_dim=flags.coupling_dim)

    def fuse_modalities(self, enc_mods: Mapping[str, VQEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsVQ:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        # batch_size is not always equal to flags.batch_size
        batch_size = enc_mods[[mod_str for mod_str in enc_mods][0]].enc_mod.shape[0]

        # transformed enc mods
        transformed_enc_mods = {mod_key: self.flow(enc_mod.enc_mod)
                                for
                                mod_key, enc_mod in enc_mods.items() if len(batch_mods) > 1}

        z_joint = torch.Tensor().to(self.flags.device)
        subsets = {}
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                z_Gf = enc_mods[s_key].enc_mod
                z_joint = torch.cat([z_joint, z_Gf])
                subsets[s_key] = z_Gf

            else:
                # sum of random variables
                subset_tf_enc_mods = torch.stack([transformed_enc_mods[mod.name][0] for mod in self.subsets[s_key]])
                z_Gf = subset_tf_enc_mods.mean(dim=0)

                # calculate inverse flow
                zss, log_det_J = self.flow.rev(z_Gf)

                z_joint = torch.cat([z_joint, zss])
                subsets[s_key] = zss

        joint_embedding = JointEmbeddingVQ(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsVQ(joint_embedding=joint_embedding, subsets=subsets)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        in_channels = 1
        self._encoder = Encoder(in_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def train():
    batch_size = 256
    num_training_updates = 15000

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 64
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay).to(device)

    model(torch.rand((1, 1, 128, 128), device=device))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    @dataclass
    class Flags:
        dir_data: Path = Path('/mnt/data/hendrik/mimic_scratch')
        mods: list = 'F'
        img_size: tuple = 128
        feature_extractor_img: str = 'resnet'
        undersample_dataset: bool = False
        len_sequence: int = 128
        word_min_occ: int = 3

    flags = Flags
    labels = get_str_labels(False)

    d_train = Mimic(flags, labels, split='train')
    d_eval = Mimic(flags, labels, split='eval')
    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True,
                              num_workers=8, drop_last=True)
    val_loader = DataLoader(d_eval, batch_size=batch_size, shuffle=False,
                            num_workers=8, drop_last=True)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for epoch in tqdm(range(150)):
        for i, (batch_d, _) in enumerate(train_loader):
            data = batch_d['PA'].to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i + 1) % 100 == 0:
                print('%d iterations' % (i + 1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                print()

        if (epoch + 1) % 1 == 0:
            model.eval()

            (valid_originals, _) = next(iter(val_loader))
            valid_originals = valid_originals['PA'].to(device)

            vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
            _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
            valid_reconstructions = model._decoder(valid_quantize)

            (train_originals, _) = next(iter(val_loader))
            train_originals = train_originals['PA'].to(device)
            _, train_reconstructions, _, _ = model._vq_vae(train_originals)

            show(make_grid(valid_reconstructions.cpu().data) + 0.5, )
            show(make_grid(valid_originals.cpu() + 0.5))


if __name__ == '__main__':
    # todo create a mimicimg class for VQ encoder and ecoder
    train()
