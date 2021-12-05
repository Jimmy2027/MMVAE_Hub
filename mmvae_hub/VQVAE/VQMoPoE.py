import typing
from typing import Mapping

import torch

from mmvae_hub.VQVAE.VQVAE import VQVAE
from mmvae_hub.VQVAE.VqVaeDataclasses import VQEncMod, JointLatentsVQ, JointEmbeddingVQ
from mmvae_hub.utils.fusion_functions import subsets_from_batchmods


class VQMoPoE(VQVAE):
    def __init__(self, exp, flags, modalities, subsets):
        super().__init__(exp, flags, modalities, subsets)

    def fuse_modalities(self, enc_mods: Mapping[str, VQEncMod],
                        batch_mods: typing.Iterable[str]) -> JointLatentsVQ:
        """
        Create a subspace for all the combinations of the encoded modalities by combining them.
        """
        batch_subsets = subsets_from_batchmods(batch_mods)

        z_joint = torch.Tensor().to(self.flags.device)
        subsets = {}
        for s_key in batch_subsets:

            if len(self.subsets[s_key]) == 1:
                z_sub = enc_mods[s_key].enc_mod
                z_joint = torch.cat([z_joint, z_sub])
                subsets[s_key] = z_sub

            else:
                # sum of random variables
                subset_enc_mods = torch.stack([enc_mods[mod.name].enc_mod for mod in self.subsets[s_key]])

                # geometric mean
                z_geom_mean = subset_enc_mods.log().mean(dim=0).exp()

                z_joint = torch.cat([z_joint, z_geom_mean])
                subsets[s_key] = z_geom_mean

        joint_embedding = JointEmbeddingVQ(embedding=z_joint, mod_strs=[k for k in batch_subsets])

        return JointLatentsVQ(joint_embedding=joint_embedding, subsets=subsets)
