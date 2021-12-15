import typing
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from modun.download_utils import download_zip_from_url
from torch import Tensor, nn

from mmvae_hub.mimic.modalities.MimicIMG import LM_
from mmvae_hub.modalities.ModalityIMG import ModalityIMG


class Residual1D(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual1D, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack1D(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack1D, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual1D(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class VQEncoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
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
        super(VQEncoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)
        return self._pre_vq_conv(x)


class VQDecoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(VQDecoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=1,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class MimicImgVQ(ModalityIMG):
    def __init__(self, data_size, flags, name, labels, rec_weight, plot_img_size):
        super().__init__(data_size, flags, name)
        self.labels = labels

        self.labels = labels
        self.gen_quality_eval = True
        self.file_suffix = '.png'

        self.encoder = VQEncoder(in_channels=1, num_hiddens=flags.img_size,
                                 num_residual_layers=flags.num_residual_layers,
                                 num_residual_hiddens=flags.num_residual_hiddens,
                                 embedding_dim=flags.embedding_dim).to(flags.device)
        self.decoder = VQDecoder(flags.embedding_dim,
                                 num_hiddens=flags.img_size,
                                 num_residual_layers=flags.num_residual_layers,
                                 num_residual_hiddens=flags.num_residual_hiddens).to(flags.device)

        self.rec_weight = rec_weight

        self.plot_img_size = plot_img_size

        self.clf_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.clf = self.get_clf()

    def get_clf(self):
        if self.flags.use_clf:
            clf_name_mapping = {'PA': 'pa', 'Lateral': 'lat'}
            # finding the directory of the classifier
            img_clf_path = Path(
                __file__).parent.parent / f'classifiers/state_dicts/{clf_name_mapping[self.name]}_clf_{self.flags.img_size}.pth'
            if not img_clf_path.exists():
                download_zip_from_url(
                    url='http://jimmy123.hopto.org:2095/nextcloud/index.php/s/GTc8pYiDKrq35ky/download',
                    dest_folder=img_clf_path.parent.parent, verbose=True)
            lightning_module = LM_(str_labels=self.labels, transforms=self.clf_transforms)
            lightning_module.model.load_state_dict(
                torch.load(img_clf_path, map_location=self.flags.device))
            return lightning_module.to(self.flags.device)

    def plot_data_single_img(self, d: Tensor):
        return plt.imshow(self.plot_data(d.squeeze(dim=0)).cpu().detach().squeeze(), cmap='gray')


class VQMimicPA(MimicImgVQ):
    def __init__(self, flags, labels: typing.Iterable[str], rec_weight, plot_img_size):
        data_size = torch.Size((1, flags.img_size, flags.img_size))
        super().__init__(data_size=data_size, flags=flags, name='PA', labels=labels, rec_weight=rec_weight,
                         plot_img_size=plot_img_size)


class VQMimicLateral(MimicImgVQ):
    def __init__(self, flags, labels: typing.Iterable[str], rec_weight, plot_img_size):
        data_size = torch.Size((1, flags.img_size, flags.img_size))
        super().__init__(data_size=data_size, flags=flags, name='Lateral', labels=labels, rec_weight=rec_weight,
                         plot_img_size=plot_img_size)


if __name__ == '__main__':
    # enc = VQEncoder(in_channels=1, num_residual_hiddens=32, embedding_dim=128, num_residual_layers=4, num_hiddens=32)
    dec = VQDecoder(in_channels=64, num_residual_hiddens=32, num_residual_layers=4, num_hiddens=32)
    # out = enc(torch.rand((12, 1, 128, 128)))
    out = dec(torch.rand((12, 64, 32, 32)))
    print('bruh')
    print(out.shape)
    #
    # img_clf_path = Path(
    #     __file__).parent.parent / f'classifiers/state_dicts/pa_clf_128.pth'
    # if not img_clf_path.exists():
    #     download_zip_from_url(
    #         url='http://jimmy123.hopto.org:2095/nextcloud/index.php/s/GTc8pYiDKrq35ky/download',
    #         dest_folder=img_clf_path.parent.parent, verbose=True)
