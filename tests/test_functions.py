import tempfile

import torch

from mmvae_hub.base.utils.Dataclasses import *
from tests.utils import set_me_up


def test_fuse_modalities():
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname)
        model = mst.mm_vae
        flags = mst.flags
        rand_latent = torch.rand(flags.batch_size, flags.class_dim).float()
        enc_mods = {k: BaseEncMod(latents_class=Distr(mu=rand_latent, logvar=rand_latent)) for k in mst.modalities}

        joint_latent = model.fuse_modalities(enc_mods, batch_mods=[k for k in mst.modalities])


if __name__ == '__main__':
    test_fuse_modalities()
