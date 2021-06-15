import textwrap
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from torchvision import transforms

from mmvae_hub.modalities import BaseModality
from mmvae_hub.modalities.utils import get_likelihood
from mmvae_hub.networks.text.ConvNetworkTextClf import ClfText
from mmvae_hub.networks.text.ConvNetworksTextMimic import EncoderText, DecoderText
from mmvae_hub.utils.plotting.save_samples import write_samples_text_to_file
from mmvae_hub.utils.utils import get_clf_path


class MimicText(BaseModality):
    def __init__(self, flags, labels: Iterable[str], rec_weight, plot_img_size, wordidx2word):
        super().__init__(flags, name='text')
        self.labels = labels
        self.likelihood_name = 'categorical'
        self.len_sequence = flags.len_sequence
        self.data_size = torch.Size((flags.vocab_size, self.len_sequence))
        self.font = ImageFont.truetype(str(Path(__file__).parent.parent / 'FreeSerif.ttf'), 20)
        self.gen_quality_eval = False
        self.file_suffix = '.txt'

        self.encoder = EncoderText(self.flags, self.flags.style_text_dim).to(flags.device)
        self.decoder = DecoderText(self.flags, self.flags.style_text_dim).to(flags.device)

        self.likelihood: torch.distributions = get_likelihood(self.likelihood_name)
        self.rec_weight = rec_weight
        self.plot_img_size = plot_img_size

        self.wordidx2word = wordidx2word

    def save_data(self, exp, d, fn, args):
        write_samples_text_to_file(self.tensor_to_text(exp, d.unsqueeze(0)), fn)

    def plot_data(self, d):
        return self.text_to_pil(d.unsqueeze(0), self.plot_img_size, self.font)

    def calc_log_prob(self, out_dist: torch.distributions, target: torch.Tensor, norm_value: int):
        target = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=self.flags.vocab_size)
        return BaseModality.calc_log_prob(out_dist, target, norm_value)

    def set_clf(self):
        model_clf = ClfText(self.flags, self.labels)
        clf_path = get_clf_path(self.flags.dir_clf,
                                self.flags.clf_save_m3 + f'vocabsize_{self.flags.vocab_size}{"_bin_label" if self.flags.binary_labels else ""}')

        model_clf.load_state_dict(torch.load(clf_path, map_location=self.flags.device))
        return model_clf.to(self.flags.device)

    def seq2text(self, seq: Iterable[int]) -> List[str]:
        """
        seg: list of indices
        """
        return [
            self.wordidx2word[str(int(seq[j]))]
            for j in range(len(seq))
        ]

    def tensor_to_text(self, gen_t: torch.Tensor, one_hot=True) -> Union[List[List[str]], List[str]]:
        """
        Converts a one hot encoded tensor or an array of indices to sentences
        gen_t: tensor of shape (bs, length_sent, num_features) if one_hot else (bs, length_sent)
        one_hot: if one_hot is True, gen_t needs to be a one-hot-encoded matrix. The maximum along every axis is taken
        to create a list of indices.
        """
        gen_t = gen_t.cpu().data.numpy()
        if one_hot:
            gen_t = np.argmax(gen_t, axis=-1)
            gen_t: np.ndarray
        if len(gen_t.shape) == 1:
            return self.seq2text(gen_t)
        decoded_samples = []
        for i in range(len(gen_t)):
            decoded = self.seq2text(gen_t[i])
            decoded_samples.append(decoded)
        return decoded_samples

    def text_to_pil(self, t, imgsize, font, w=128, h=256, linewidth: int = 27, max_nbr_lines: int = 10,
                    text_cleanup=True):
        """
        text_cleanup: if true, remove padding tokens in text for the plot.
        linewidth: max number of characters per line on the image.
        max_nb_lines: maximum number of lines that will fit on the image. If the wrapped text contains mor lines, the rest
        will be left out.
        """

        blank_img = torch.ones([imgsize[0], w, h])
        pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        one_hot = len(t.shape) > 2
        sep = ' '
        text_sample = self.tensor_to_text(t, one_hot=one_hot)[0]

        if text_cleanup:
            text_sample = [word for word in text_sample if word != '<pad>']

        text_sample = sep.join(text_sample).translate({ord('*'): None}).replace(' .', '.')

        lines = textwrap.wrap(text_sample, width=linewidth)
        lines = lines[:max_nbr_lines]
        lines = '\n'.join(lines)

        draw.multiline_text((10, 10), lines, font=font, fill=(0, 0, 0))

        if imgsize[0] == 3:
            return transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                        Image.ANTIALIAS))
        else:
            return transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                        Image.ANTIALIAS).convert('L'))
