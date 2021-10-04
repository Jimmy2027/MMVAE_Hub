import textwrap
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from mmvae_hub.utils.setup.flags_utils import get_config_path

from mmvae_hub.utils.utils import json2dict
from modun.download_utils import download_from_url
from torch.distributions import OneHotCategorical
from torchvision import transforms

from mmvae_hub.celeba.networks.ConvNetworkTextClfCelebA import ClfText
from mmvae_hub.celeba.networks.ConvNetworksTextCelebA import EncoderText, DecoderText
from mmvae_hub.modalities import BaseModality
from mmvae_hub.utils.plotting.save_samples import write_samples_text_to_file
from mmvae_hub.utils.text import tensor_to_text


class CelebaText(BaseModality):
    def __init__(self, flags, len_sequence, alphabet):
        name = 'text'
        super().__init__(flags, name)
        self.flags = flags
        self.name = name
        self.px_z = OneHotCategorical

        self.alphabet = alphabet

        self.num_features = len(self.alphabet)

        self.len_sequence = len_sequence
        self.data_size = torch.Size([len_sequence])
        self.plot_img_size = torch.Size((3, 64, 64))
        self.font = self.get_font()
        self.gen_quality_eval = False
        self.file_suffix = '.txt'

        self.encoder = EncoderText(self.flags, self.num_features).to(flags.device)
        self.decoder = DecoderText(self.flags, self.num_features).to(flags.device)

        self.clf = self.get_clf()

    def get_font(self):
        font_path = Path(__file__).parent.parent.parent / 'modalities/text/FreeSerif.ttf'
        return ImageFont.truetype(str(font_path), 38)

    def save_data(self, d, fn, args):
        write_samples_text_to_file(tensor_to_text(self.alphabet,
                                                  d.unsqueeze(0)),
                                   fn)

    def plot_data(self, d):
        return self.text_to_pil(t=d.unsqueeze(0), imgsize=self.plot_img_size,
                                alphabet=self.alphabet, font=self.font,
                                w=256, h=256, linewidth=16)

    def get_clf(self):
        if self.flags.use_clf:
            img_clf_path = self.flags.dir_clf / 'clf_celeba_text.pth'
            if not img_clf_path.exists():
                download_from_url(
                    url='https://www.dropbox.com/sh/lx8669lyok9ois6/AACaBy1YNNq3ebh149k_EXrca/trained_classifiers/trained_clfs_celeba/clf_m2?dl=1',
                    dest_path=img_clf_path, verbose=True)

            clf = ClfText(self.flags, num_features=self.num_features)
            clf.load_state_dict(torch.load(img_clf_path, map_location=self.flags.device))

            return clf.to(self.flags.device)

    def calc_likelihood(self, style_embeddings, class_embeddings, unflatten: Tuple = None):
        if unflatten:
            return self.px_z(self.decoder(None, class_embeddings)[0].unflatten(0, unflatten))

        text_hat = self.decoder(style_embeddings, class_embeddings)[0]
        # ok = self.px_z.arg_constraints["probs"].check(text_hat)
        # bad_elements = text_hat[~ok]
        # print(bad_elements)
        return self.px_z(text_hat, validate_args=False)

    def batch_text_to_onehot(self, batch_text, vocab_size: int):
        """In the mnistsvhntext dataset the text is already in one hot format"""
        return batch_text

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


if __name__ == '__main__':

    config = json2dict(get_config_path(dataset='celeba'))

    img_clf_path = Path(config['dir_clf']) / 'clf_celeba_img.pth'

    if not img_clf_path.exists():
        download_from_url(
            url='https://www.dropbox.com/sh/lx8669lyok9ois6/AACaBy1YNNq3ebh149k_EXrca/trained_classifiers/trained_clfs_celeba/clf_m2?dl=1',
            dest_path=img_clf_path, verbose=True)
    print("Done.")
