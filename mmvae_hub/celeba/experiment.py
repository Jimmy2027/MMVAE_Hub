import json
import random
from pathlib import Path

import PIL.Image as Image
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchvision import transforms

from mmvae_hub.base.BaseExperiment import BaseExperiment
from mmvae_hub.celeba.CelebADataset import CelebaDataset
from mmvae_hub.celeba.metrics import CelebAMetrics
from mmvae_hub.celeba.modalities.celebaImg_ import CelebaImg_
from mmvae_hub.celeba.modalities.celebaText import CelebaText
from mmvae_hub.mimic.modalities.MimicIMG import MimicImg
LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
          'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
          'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
          'Mouth_Slightly_Open', 'Mustache',
          'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
          'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
          'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
          'Wearing_Earrings', 'Wearing_Hat',
          'Wearing_Lipstick', 'Wearing_Necklace',
          'Wearing_Necktie', 'Young']


class CelebaExperiment(BaseExperiment):
    def __init__(self, flags):
        super(CelebaExperiment, self).__init__(flags)
        self.labels = LABELS
        self.flags = flags
        self.dataset = 'celeba'

        alphabet_path = Path(__file__).parent.parent / ('modalities/text/alphabet.json')
        with open(alphabet_path) as alphabet_file:
            self.alphabet = str(''.join(json.load(alphabet_file)))

        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        print(self.mm_vae)

        self.optimizer = None
        self.rec_weights = self.set_rec_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = average_precision_score
        self.paths_fid = self.set_paths_fid()

        self.metrics = CelebAMetrics

    def set_modalities(self):
        # mod1 = CelebaImg(self.flags)
        mod2 = CelebaText(flags=self.flags,
                          len_sequence=self.flags.len_sequence,
                          alphabet=self.alphabet)
        mod1 = MimicImg(flags=self.flags, name='img', labels=self.labels, rec_weight=1.,
                        plot_img_size=torch.Size((3, 64, 64)), data_size=torch.Size((3, 64, 64)))
        # mod1 = CelebaImg_(flags=self.flags, name='img')
        # mod1 = MNIST(self.flags, 'img')
        # mod2 = Text(self.flags, self.alphabet)

        return {mod1.name: mod1, mod2.name: mod2}
        # return {mod1.name: mod1}

    def get_transform_celeba(self):
        offset_height = (218 - self.flags.crop_size_img) // 2
        offset_width = (178 - self.flags.crop_size_img) // 2
        crop = lambda x: x[:, offset_height:offset_height + self.flags.crop_size_img,
                         offset_width:offset_width + self.flags.crop_size_img]
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Lambda(crop),
                                   transforms.ToPILImage(),
                                   transforms.Resize(size=(self.flags.img_size,
                                                           self.flags.img_size),
                                                     interpolation=Image.BICUBIC),
                                   transforms.ToTensor()])

    def set_dataset(self):
        transform = self.get_transform_celeba()
        d_train = CelebaDataset(self.flags, self.alphabet, partition=0, transform=transform)
        d_eval = CelebaDataset(self.flags, self.alphabet, partition=1, transform=transform)
        self.dataset_train = d_train
        self.dataset_test = d_eval

    def set_rec_weights(self) -> None:
        ref_mod_d_size = self.modalities['img'].data_size.numel() / 3
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            mod.rec_weight = float(ref_mod_d_size / numel_mod)

    def get_prediction_from_attr(self, values):
        return values.ravel()

    def get_prediction_from_attr_random(self, values, index=None):
        return values[:, index] > 0.5

    def eval_label(self, values, labels, index=None):
        pred = values[:, index]
        gt = labels[:, index]
        try:
            ap = self.eval_metric(gt, pred)
        except ValueError:
            ap = 0.0
        return ap

    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__()
        samples = []
        for _ in range(10):
            ix = np.random.randint(0, len(self.dataset_test.img_names))
            sample, target = self.dataset_test.__getitem__(random.randint(0, n_test))
            for k, key in enumerate(sample):
                sample[key] = sample[key].to(self.flags.device)
            samples.append(sample)
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def set_style_weights(self):
        return {'img': 0, 'text': 0}
