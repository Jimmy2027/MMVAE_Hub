import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union, Tuple

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.distributions import OneHotCategorical
from torchvision import transforms
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from mmvae_hub.modalities import BaseModality
from mmvae_hub.networks.text.ConvNetworksTextMimic import EncoderText, DecoderText
from mmvae_hub.utils.plotting.save_samples import write_samples_text_to_file


class MimicText(BaseModality):
    def __init__(self, flags, labels: Iterable[str], rec_weight, plot_img_size, wordidx2word):
        super().__init__(flags, name='text')
        self.labels = labels

        self.len_sequence = flags.len_sequence
        self.data_size = torch.Size((flags.vocab_size, self.len_sequence))
        self.font = ImageFont.truetype(str(Path(__file__).parent.parent / 'FreeSerif.ttf'), 24)
        self.gen_quality_eval = False
        self.file_suffix = '.txt'

        self.encoder = EncoderText(self.flags, self.flags.style_text_dim).to(flags.device)
        self.decoder = DecoderText(self.flags, self.flags.style_text_dim).to(flags.device)

        self.px_z = OneHotCategorical
        self.rec_weight = rec_weight
        # self.plot_img_size = torch.Size([1, 256, 128])
        self.plot_img_size = torch.Size([1, 128, 128])

        self.wordidx2word = wordidx2word

        self.clf = self.get_clf()

        self.vocab_size = flags.vocab_size
        self.num_features = self.vocab_size

    def save_data(self, d, fn, args):
        write_samples_text_to_file(self.tensor_to_text(gen_t=d.unsqueeze(0)), fn)

    def plot_data(self, d):
        return self.text_to_pil(d.unsqueeze(0), self.plot_img_size, self.font)

    def plot_data_single_img(self, d: Tensor):
        return plt.imshow(self.plot_data(d.squeeze(0)).squeeze(), cmap='gray')

    def calc_log_prob(self, out_dist: torch.distributions, target: torch.Tensor, norm_value: int):
        target = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=self.flags.vocab_size)
        return BaseModality.calc_log_prob(out_dist, target, norm_value)

    def get_clf(self):
        if self.flags.use_clf:
            clf = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                      num_labels=len(self.labels)).to(
                self.flags.device)
            text_clf_path = Path(__file__).parent.parent / 'classifiers/state_dicts/text_clf.pth'
            clf.load_state_dict(torch.load(text_clf_path, map_location=self.flags.device))
            return TextClf(self, clf).to(self.flags.device)

    def calc_likelihood(self, class_embeddings, unflatten: Tuple = None):
        "Calculate px_z"
        if unflatten:
            logits = self.decoder(class_embeddings)[0]
            return self.px_z(logits=logits.unflatten(0, unflatten), validate_args=False)

        return self.px_z(logits=self.decoder(class_embeddings)[0], validate_args=False)

    def log_likelihood(self, px_z, batch_sample):
        return px_z.log_prob(self.batch_text_to_onehot(batch_sample))

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

    def batch_text_to_onehot(self, batch_text, vocab_size=None):
        return torch.nn.functional.one_hot(batch_text.to(torch.int64),
                                           num_classes=self.vocab_size)


class TextClf(nn.Module):
    def __init__(self, text_mod: MimicText, clf):
        super().__init__()
        self.text_mod = text_mod
        self.clf = clf
        tokenizer_path = Path(__file__).parent.parent / 'classifiers/tokenizer'
        if not tokenizer_path.exists():
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            tokenizer.save_pretrained(tokenizer_path)
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer

    def forward(self, x):
        x_ = [' '.join(sent) for sent in self.text_mod.tensor_to_text(x)]

        item = {key: torch.tensor(val).to(x.device) for key, val in
                self.tokenizer(x_, return_tensors="pt", padding=True, truncation=True, max_length=256).items()}
        return self.clf(**item).logits


if __name__ == '__main__':
    # clf = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
    #                                                           num_labels=3)
    # text_clf_path = Path(__file__).parent.parent / 'classifiers/state_dicts/text_clf.pth'
    # clf.load_state_dict(torch.load(text_clf_path))

    @dataclass
    class Flags:
        device = 'cpu'
        vocab_size = 2000
        len_sequence = 128
        style_text_dim = 0
        DIM_text = 128
        class_dim = 512
        use_clf = False

    wordidx2word = {'27': 'HelloHelloHelloHello'}

    flags = Flags()
    textmod = MimicText(flags = flags, labels = [''], rec_weight=1, plot_img_size=None, wordidx2word=wordidx2word)
    t = torch.tensor([27 for _ in range(128)]).unsqueeze(dim=0)
    # plt.imshow(textmod.text_to_pil(t= t, imgsize = textmod.plot_img_size, font = textmod.font).squeeze())
    textmod.plot_data_single_img(d= t)
    plt.show()
