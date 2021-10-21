import os
from pathlib import Path

import PIL.Image as Image
import pandas as pd
import torch
from modun.download_utils import download_zip_from_url
from modun.file_io import json2dict
from torch.utils.data import Dataset

from mmvae_hub import log
from mmvae_hub.utils.setup.flags_utils import get_config_path
from mmvae_hub.utils.text import one_hot_encode


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, args, alphabet, partition=0, transform=None):
        self.dir_dataset_base = args.dir_data

        if not self.dir_dataset_base.exists():
            log.info(
                f'Dataset not found under {self.dir_dataset_base}. Parent directory contains: '
                f'{list(self.dir_dataset_base.parent)}')
            download_zip_from_url(
                url='https://www.dropbox.com/sh/lx8669lyok9ois6/AACCZqDiZuv0Q8RA3Qmwrwnca/celeba_data.zip?dl=1',
                dest_folder=Path(self.dir_dataset_base).parent, verbose=True)

        filename_text = self.dir_dataset_base / ('list_attr_text_' + str(args.len_sequence).zfill(3) + '_' + str(
            args.random_text_ordering) + '_' + str(args.random_text_startindex) + '_celeba.csv')
        filename_partition = os.path.join(self.dir_dataset_base, 'list_eval_partition.csv')
        filename_attributes = os.path.join(self.dir_dataset_base, 'list_attr_celeba.csv')

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition)
        df_attributes = pd.read_csv(filename_attributes)

        self.args = args
        self.img_dir = os.path.join(self.dir_dataset_base, 'img_align_celeba')
        self.txt_path = filename_text
        self.attrributes_path = filename_attributes
        self.partition_path = filename_partition

        self.alphabet = alphabet
        self.img_names = df_text.loc[df_partition['partition'] == partition]['image_id'].values
        self.attributes = df_attributes.loc[df_partition['partition'] == partition]
        self.labels = df_attributes.loc[
            df_partition['partition'] == partition].values  # atm, i am just using blond_hair as labels
        self.y = df_text.loc[df_partition['partition'] == partition]['text'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
        text_str = one_hot_encode(self.args.len_sequence, self.alphabet, self.y[index])
        label = torch.from_numpy((self.labels[index, 1:] > 0).astype(int)).float()
        # img = torch.rand((self.args.image_channels, 64, 64))
        # text_str = torch.ones((8, 71))
        # img = torch.ones((1, 28, 28))
        # text_str = torch.rand((256, 71))
        sample = {'img': img, 'text': text_str}
        # sample = {'img': img}
        return sample, label

    def __len__(self):
        return self.y.shape[0]

    def get_text_str(self, index):
        return self.y[index]


if __name__ == '__main__':

    config = json2dict(get_config_path(dataset='celeba'))

    dir_data = Path(config['dir_data']).expanduser()

    if not dir_data.exists():
        download_zip_from_url(
            url='https://www.dropbox.com/sh/lx8669lyok9ois6/AACCZqDiZuv0Q8RA3Qmwrwnca/celeba_data.zip?dl=1',
            dest_folder=dir_data.parent, verbose=True)
    print("Done.")
