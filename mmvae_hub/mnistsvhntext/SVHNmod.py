import os

from PIL import Image
import torch
from mmvae_hub.mnistsvhntext.networks.ConvNetworkImgClfSVHN import ClfImgSVHN
from modun.download_utils import download_zip_from_url

from mmvae_hub.mnistsvhntext.networks.ConvNetworksImgSVHN import EncoderSVHN, DecoderSVHN
from torchvision import transforms

from mmvae_hub.polymnist.PolymnistMod import PolymnistMod


class SVHN(PolymnistMod):
    def __init__(self, flags, name):
        super().__init__(flags=flags, name=name)
        self.rec_weight = 1.
        self.plot_img_size = torch.Size((3, 28, 28))
        self.transform_plot = self.get_plot_transform()

        self.encoder = EncoderSVHN(flags).to(flags.device)
        self.decoder = DecoderSVHN(flags).to(flags.device)

    def plot_data(self, d):
        return self.transform_plot(d.squeeze(0).cpu()).cuda().unsqueeze(0)

    def get_plot_transform(self):
        transf = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=list(self.plot_img_size)[1:],
                                                       interpolation=Image.BICUBIC),
                                     transforms.ToTensor()])
        return transf


    def get_clf(self):
        if self.flags.use_clf:
            dir_clf = self.flags.dir_clf
            if not dir_clf.exists():
                download_zip_from_url(
                    url='https://www.dropbox.com/sh/lx8669lyok9ois6/AADM7Cs_QReijyo2kF8xzWqua/trained_classifiers/trained_clfs_mst?dl=1',
                    dest_folder=dir_clf)
            model_clf =  ClfImgSVHN()
            model_clf.load_state_dict(
                torch.load(os.path.join(self.flags.dir_clf, f"clf_m2"),
                           map_location=self.flags.device))

            return model_clf.to(self.flags.device)