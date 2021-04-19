import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


class MMNISTDataset(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, dir_data: Path, transform=None, target_transform=None, num_modalities: int = 5):
        """
            Args:
                unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                    correspond by index. Therefore the numbers of samples of all datapaths should match.
                transform: tranforms on colored MNIST digits.
                target_transform: transforms on labels.
                num_modalities (int):  number of modalities.
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.unimodal_datapaths = sorted(glob.glob(str(dir_data / 'm*')))[:num_modalities]
        self.transform = transform
        self.target_transform = target_transform

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        for dp in self.unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

    @staticmethod
    def create_mmnist_dataset(savepath: Path, backgroundimagepath:Path, num_modalities, train):
        """Create the Multimodal MNIST Dataset under 'savepath' given a directory of background images.
        
            Args:
                savepath : path to directory that the dataset will be written to. Will be created if it does not
                    exist.
                backgroundimagepath : path to a directory filled with background images. One background images is
                    used per modality.
                num_modalities (int): number of modalities to create.
                train (bool): create the dataset based on MNIST training (True) or test data (False).
        
        """

        # load MNIST data
        mnist = datasets.MNIST("/tmp", train=train, download=True, transform=None)

        # load background images
        background_filepaths = sorted(
            glob.glob(os.path.join(backgroundimagepath, "*.jpg")))  # TODO: handle more filetypes
        print("\nbackground_filepaths:\n", background_filepaths, "\n")
        if num_modalities > len(background_filepaths):
            raise ValueError("Number of background images must be larger or equal to number of modalities")
        background_images = [Image.open(fp) for fp in background_filepaths]

        # create the folder structure: savepath/m{1..num_modalities}
        for m in range(num_modalities):
            unimodal_path = os.path.join(savepath, "m%d" % m)
            if not os.path.exists(unimodal_path):
                os.makedirs(unimodal_path)
                print("Created directory", unimodal_path)

        # create random pairing of images with the same digit label, add background image, and save to disk
        cnt = 0
        for digit in range(10):
            ixs = (mnist.targets == digit).nonzero()
            for m in range(num_modalities):
                ixs_perm = ixs[torch.randperm(len(ixs))]  # one permutation per modality and digit label
                for i, ix in enumerate(ixs_perm):
                    # add background image
                    new_img = MMNISTDataset._add_background_image(background_images[m], mnist.data[ix])
                    # save as png
                    filepath = os.path.join(savepath, "m%d/%d.%d.png" % (m, i, digit))
                    save_image(new_img, filepath)
                    # log the progress
                    cnt += 1
                    if cnt % 10000 == 0:
                        print("Saved %d/%d images to %s" % (cnt, len(mnist) * num_modalities, savepath))
        assert cnt == len(mnist) * num_modalities

    @staticmethod
    def _add_background_image(background_image_pil, mnist_image_tensor, change_colors=False):

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 128).type(torch.bool)  # NOTE: mnist is _not_ normalized to [0, 1]

        # squeeze away color channel
        if img_binarized.ndimension() == 2:
            pass
        elif img_binarized.ndimension() == 3:
            img_binarized = img_binarized.squeeze(0)
        else:
            raise ValueError("Unexpected dimensionality of MNIST image:", img_binarized.shape)

        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 28)
        y_c = np.random.randint(0, background_image_pil.size[1] - 28)
        new_img = background_image_pil.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img)
        if change_colors:  # Change color distribution
            for j in range(3):
                new_img[:, :, j] = (new_img[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        new_img[:, img_binarized] = 1 - new_img[:, img_binarized]

        return new_img

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        files = [self.file_paths[dp][index] for dp in self.unimodal_datapaths]
        images = [Image.open(files[m]) for m in range(self.num_modalities)]
        labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]

        # transforms
        if self.transform:
            images = [self.transform(img) for img in images]
        if self.target_transform:
            labels = [self.transform(label) for label in labels]

        images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        return images_dict, labels[0]  # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files


class ToyMMNISTDataset(Dataset):
    """Toy MMNIST dataset for testing purposes."""

    def __init__(self, num_modalities: int = 5):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_files = 420

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        images_dict = {"m%d" % m: torch.rand(3, 28, 28).float() for m in range(self.num_modalities)}
        return images_dict, random.randint(0, 10)

    def __len__(self):
        return self.num_files


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--num-modalities', type=int, default=5)
    # parser.add_argument('--savepath-train', type=str, required=True)
    # parser.add_argument('--savepath-test', type=str, required=True)
    # parser.add_argument('--backgroundimagepath', type=str, required=True)
    # args = parser.parse_args()  # use vars to convert args into a dict
    # print("\nARGS:\n", args)
    from dataclasses import dataclass
    from pathlib import Path
    from mmvae_hub.base.utils.flags_utils import get_config_path, json2dict

    config = json2dict(get_config_path())


    @dataclass
    class Args:
        savepath_train: Path = Path(config['dir_data']) / 'train'
        savepath_test: Path = Path(config['dir_data']) / 'test'
        backgroundimagepath: Path = Path(__file__).parent / 'mmnist_background_images'
        num_modalities: int = 5


    args = Args()
    # create dataset
    MMNISTDataset.create_mmnist_dataset(args.savepath_train, args.backgroundimagepath, args.num_modalities, train=True)
    MMNISTDataset.create_mmnist_dataset(args.savepath_test, args.backgroundimagepath, args.num_modalities, train=False)
    print("Done.")
