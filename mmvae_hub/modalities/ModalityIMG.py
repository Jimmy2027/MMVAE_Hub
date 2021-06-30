import PIL.Image as Image
from torch import Tensor
from torchvision import transforms

from mmvae_hub.modalities import BaseModality
from mmvae_hub.utils.plotting.save_samples import write_samples_img_to_file


class ModalityIMG(BaseModality):
    def __init__(self, data_size, flags, name):
        super().__init__(flags, name)
        self.data_size = data_size

    def save_data(self, exp, d, fn, args):
        img_per_row = args['img_per_row']
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d: Tensor):
        if d.shape != self.plot_img_size:
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(size=self.plot_img_size[1:], interpolation=Image.BICUBIC),
                                            transforms.ToTensor()])
            d = transform(d.cpu())
        return d.repeat(1, 1, 1, 1)

    def calc_likelihood(self, style_embeddings, class_embeddings):
        return self.likelihood(**self.decoder(style_embeddings, class_embeddings))
