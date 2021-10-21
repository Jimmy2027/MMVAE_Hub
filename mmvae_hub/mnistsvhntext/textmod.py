import os
import textwrap
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from modun.download_utils import download_zip_from_url
from torch.distributions import OneHotCategorical
from torchvision import transforms

from mmvae_hub.mnistsvhntext.networks.ConvNetworkTextClf import ClfText
from mmvae_hub.mnistsvhntext.networks.ConvNetworksTextMNIST import EncoderText, DecoderText
from mmvae_hub.modalities import BaseModality
from mmvae_hub.utils.plotting.save_samples import write_samples_text_to_file
from mmvae_hub.utils.text import tensor_to_text


class Text(BaseModality):
    def __init__(self, flags, alphabet):
        super().__init__(flags, name='text')
        self.alphabet = alphabet
        self.rec_weight = 1.
        self.px_z = OneHotCategorical
        self.font = self.get_font()

        self.len_sequence = flags.len_sequence
        self.data_size = torch.Size([self.len_sequence]);
        self.plot_img_size = torch.Size((3, 28, 28))

        self.num_features = len(self.alphabet)

        self.encoder = EncoderText(flags).to(flags.device)
        self.decoder = DecoderText(flags).to(flags.device)

        self.clf = self.get_clf()

        self.gen_quality_eval = False;
        self.file_suffix = '.txt';

    def save_data(self, d, fn, args):
        write_samples_text_to_file(tensor_to_text(self.alphabet, d.unsqueeze(0)), fn)

    def plot_data(self, d):
        out = self.text_to_pil(d.unsqueeze(0), self.plot_img_size,
                               self.alphabet, self.font)
        return out

    def get_clf(self):
        if self.flags.use_clf:
            dir_clf = self.flags.dir_clf
            if not dir_clf.exists():
                download_zip_from_url(
                    url='https://www.dropbox.com/sh/lx8669lyok9ois6/AADM7Cs_QReijyo2kF8xzWqua/trained_classifiers/trained_clfs_mst?dl=1',
                    dest_folder=Path(self.flags.dir_clf))
            model_clf = ClfText(self.flags)
            model_clf.load_state_dict(
                torch.load(os.path.join(self.flags.dir_clf, f"clf_m3"),
                           map_location=self.flags.device))

            return model_clf.to(self.flags.device)

    def calc_likelihood(self, class_embeddings, unflatten: Tuple = None):
        if unflatten:
            return self.px_z(self.decoder(class_latent_space=class_embeddings)[0].unflatten(0, unflatten), validate_args=False)

        return self.px_z(self.decoder(class_latent_space=class_embeddings)[0], validate_args=False)

    def text_to_pil(self, t, imgsize, alphabet, font, w=128, h=128, linewidth=8):
        blank_img = torch.ones([imgsize[0], w, h]);
        pil_img = transforms.ToPILImage()(blank_img.cpu()).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        text_sample = tensor_to_text(alphabet, t)[0]
        text_sample = ''.join(text_sample).translate({ord('*'): None})
        lines = textwrap.wrap(''.join(text_sample), width=linewidth)
        y_text = h
        num_lines = len(lines);
        for l, line in enumerate(lines):
            width, height = font.getsize(line)
            draw.text((0, (h / 2) - (num_lines / 2 - l) * height), line, (0, 0, 0), font=font)
            y_text += height
        if imgsize[0] == 3:
            text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                            Image.ANTIALIAS));
        else:
            text_pil = transforms.ToTensor()(pil_img.resize((imgsize[1], imgsize[2]),
                                                            Image.ANTIALIAS).convert('L'));
        return text_pil;

    def get_font(self):
        font_path = Path(__file__).parent.parent / 'modalities/text/FreeSerif.ttf'
        return ImageFont.truetype(str(font_path), 38)

    def batch_text_to_onehot(self, batch_text, vocab_size: int):
        """In the mnistsvhntext dataset the text is already in one hot format"""
        return batch_text
