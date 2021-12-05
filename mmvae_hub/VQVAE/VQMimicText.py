from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from mmvae_hub.VQVAE.VQMimicIMG import ResidualStack, ResidualStack1D
from mmvae_hub.mimic.modalities.MimicText import MimicText
from mmvae_hub.networks.text.mmvae_text_enc import make_res_block_enc_feat_ext


class VQEncoderText(nn.Module):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 embedding_dim,
                 vocab_size: int,
                 DIM_text: int,
                 len_sequence: int):
        """

        :param in_channels: number of channels of input
        :param num_hiddens: hidden channel size
        :param num_residual_layers: number of residual layers in encoder
        :param num_residual_hiddens: hidden layer shape
        :param embedding_dim: output channels
        enc = VQEncoder(in_channels=1, num_residual_hiddens=32, embedding_dim=64, num_residual_layers=3, num_hiddens=64)

        Example:
        >>> enc = VQEncoder(in_channels=1, num_residual_hiddens=1, embedding_dim=64, num_residual_layers=3, num_hiddens=64)
        >>> out = enc(torch.rand((12, 1, 128, 128)))
        >>> print(out.shape)
        torch.Size([12, 64, 32, 32])
        """
        super(VQEncoderText, self).__init__()

        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=DIM_text,
                                           padding_idx=0)

        self._conv_1 = nn.Conv2d(in_channels=1,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

    def forward(self, inputs):
        emb = self.text_embedding(inputs.long())
        emb = F.relu(emb)
        emb = emb.unsqueeze(1)

        x = self._conv_1(emb)
        x = F.relu(x)

        x = self._conv_2(x)

        x = self._residual_stack(x)

        return self._pre_vq_conv(x)


class VQDecoderText(nn.Module):
    def __init__(self, embedding_dim):
        super(VQDecoderText, self).__init__()

        self._residual_stack = ResidualStack1D(in_channels=embedding_dim,
                                               num_hiddens=embedding_dim,
                                               num_residual_layers=3,
                                               num_residual_hiddens=embedding_dim)

        self.resblock1 = make_res_block_enc_feat_ext(embedding_dim,
                                                     3 * embedding_dim,
                                                     kernelsize=4, stride=2, padding=1, dilation=1)

        self.resblock2 = make_res_block_enc_feat_ext(3 * embedding_dim,
                                                     12 * embedding_dim,
                                                     kernelsize=4, stride=2, padding=1, dilation=1)

        self.resblock3 = make_res_block_enc_feat_ext(12 * embedding_dim,
                                                     24 * embedding_dim,
                                                     kernelsize=4, stride=2, padding=1, dilation=1)

        # self.resblock4 = make_res_block_enc_feat_ext(24 * embedding_dim,
        #                                              48 * embedding_dim,
        #                                              kernelsize=4, stride=1, padding=1, dilation=1)

        self.conv = nn.Conv1d(in_channels=24 * embedding_dim,
                              out_channels=3517,
                              kernel_size=1,
                              stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = torch.flatten(inputs, start_dim=2, end_dim=3)

        x = self._residual_stack(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        # x = self.resblock4(x)
        x = self.conv(x)
        return self.softmax(x)


class VQMimicText(MimicText):
    def __init__(self, flags, labels: Iterable[str], rec_weight, plot_img_size, wordidx2word):
        super().__init__(flags, labels, rec_weight, plot_img_size, wordidx2word)

        self.encoder = VQEncoderText(num_residual_hiddens=32, embedding_dim=128, num_residual_layers=4, num_hiddens=32,
                                     len_sequence=128, vocab_size=flags.vocab_size, DIM_text=128).to(flags.device)
        self.decoder = VQDecoderText(embedding_dim=128).to(flags.device)


if __name__ == '__main__':
    # enc = VQEncoderText(num_residual_hiddens=32, embedding_dim=128, num_residual_layers=4, num_hiddens=32,
    #                     len_sequence=128, vocab_size=1024, DIM_text=128)
    # out = enc(torch.rand((12, 128)))

    dec = VQDecoderText(embedding_dim=128)
    out = dec(torch.rand((2, 128, 32, 32)))
