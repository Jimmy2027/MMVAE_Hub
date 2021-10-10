from pathlib import Path

import torch
from PIL import Image
from mmvae_hub.utils.setup.flags_utils import get_config_path

from mmvae_hub.utils.utils import json2dict
from modun.download_utils import download_from_url
from torchvision import transforms

from mmvae_hub.celeba.networks.ConvNetworkImgClfCelebA import ClfImg
from mmvae_hub.celeba.networks.ConvNetworksImgCelebA import EncoderImg, DecoderImg
from mmvae_hub.modalities.ModalityIMG import ModalityIMG
# from mmvae_hub.networks.images.ConvNetworksImgMimic import EncoderImg, DecoderImg


class CelebaImg(ModalityIMG):
    def __init__(self, flags):
        data_size = torch.Size((3, 64, 64))
        name = 'img'
        super().__init__(data_size, flags, name)

        self.plot_img_size = torch.Size((3, 64, 64))
        self.transform_plot = self.get_plot_transform()
        self.gen_quality_eval = True
        self.file_suffix = '.png'

        self.encoder = EncoderImg(self.flags).to(flags.device)
        self.decoder = DecoderImg(self.flags).to(flags.device)

        self.clf = self.get_clf()

    def plot_data(self, d):
        return self.transform_plot(d.squeeze(0).cpu()).cuda().unsqueeze(0)

    def get_plot_transform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    size=list(self.plot_img_size)[1:], interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )

    def get_clf(self):
        if self.flags.use_clf:
            img_clf_path = self.flags.dir_clf / 'clf_celeba_img.pth'
            if not img_clf_path.exists():
                download_from_url(
                    url='https://www.dropbox.com/sh/lx8669lyok9ois6/AABxwQKXU1no5cM91eStMhfIa/trained_classifiers/trained_clfs_celeba/clf_m1?dl=1',
                    dest_path=img_clf_path, verbose=True)

            clf = ClfImg(self.flags)
            clf.load_state_dict(torch.load(img_clf_path, map_location=self.flags.device))

            return clf.to(self.flags.device)


if __name__ == '__main__':

    config = json2dict(get_config_path(dataset='celeba'))

    img_clf_path = Path(config['dir_clf']).expanduser() / 'clf_celeba_img.pth'
    img_clf_path.parent.mkdir(exist_ok=True, parents=True)
    if not img_clf_path.exists():
        print(f'img clf not found under {img_clf_path}. Parent folder contains: {list(img_clf_path.parent.iterdir())}')
        download_from_url(
            url='https://www.dropbox.com/sh/lx8669lyok9ois6/AABxwQKXU1no5cM91eStMhfIa/trained_classifiers/trained_clfs_celeba/clf_m1?dl=1',
            dest_path=img_clf_path, verbose=True)
