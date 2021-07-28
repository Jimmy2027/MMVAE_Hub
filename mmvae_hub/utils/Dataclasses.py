# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Mapping, Optional, Iterable

import torch
from torch import Tensor
from torch.autograd import Variable


@dataclass
class BaseLatents:
    enc_mods: dict
    joint: dict


@dataclass
class BaseDivergences:
    joint_div: float
    mods_div: Mapping[str, Tensor]


@dataclass
class Distr:
    mu: Tensor
    logvar: Tensor
    mod_strs: Optional[Iterable[str]] = None

    def reparameterize(self) -> Tensor:
        """
        Sample z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        std = self.logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.mu)


@dataclass
class PlanarFlowParams:
    u: Tensor
    w: Tensor
    b: Tensor


@dataclass
class EncModPlanarMixture:
    """
    zk: embedding after the kth flow. Has shape (bs, class_dim)
    """
    latents_class: Distr
    flow_params: PlanarFlowParams
    z0: Optional[Tensor] = None
    zk: Optional[Tensor] = None
    log_det_j: Optional[Tensor] = None
    latents_style: Optional[Distr] = None


@dataclass
class EncModPFoM:
    latents_class: Distr
    h: Tensor
    latents_style: Optional[Distr] = None


@dataclass
class EncModGfM:
    zk: Tensor


@dataclass
class BaseEncMod:
    # latents have shape [batch_size, class_dim]
    latents_class: Distr
    latents_style: Optional[Distr] = None


@dataclass
class JointLatents:
    fusion_subsets_keys: Iterable[str]
    joint_distr: Distr
    subsets: Mapping[str, Distr]

    def get_joint_embeddings(self):
        return self.joint_distr.reparameterize()

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].reparameterize()

    def get_q0(self, subset_key: str):
        """Return the mean of the subset."""
        if subset_key == 'joint':
            return self.get_joint_q0()
        return self.subsets[subset_key].mu

    def get_joint_q0(self):
        return self.joint_distr.mu

    def get_lreval_data(self) -> dict:
        """Get lr_data for the lr evaluation."""
        lr_data = {'q0': {}}
        for key in self.subsets:
            lr_data['q0'][key] = self.get_q0(key).cpu()

        lr_data['q0']['joint'] = self.get_joint_q0().cpu()

        return lr_data

    def get_lreval_data_(self, data_train: dict):
        """Add lr values to data_train."""
        for key in self.subsets:
            data_train['q0'][key] = torch.cat((data_train['q0'][key], self.get_q0(key).cpu()), 0)
        joint_q0 = self.get_joint_q0().cpu()
        data_train['q0']['joint'] = torch.cat((data_train['q0']['joint'], joint_q0), 0)

        return data_train


@dataclass
class JointEmbeddingFoEM:
    embedding: Tensor
    mod_strs: Iterable[str]


@dataclass
class JointLatentsFoEM:
    """Joint Latens for flow of enc mods methods."""
    joint_embedding: JointEmbeddingFoEM

    subsets: Mapping[str, Tensor]

    def get_joint_embeddings(self):
        return self.joint_embedding.embedding

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key]

    def get_zk(self, subset_key: str):
        """Return the embedding of the subset."""
        if subset_key == 'joint':
            return self.get_joint_zk()
        return self.subsets[subset_key]

    def get_joint_zk(self):
        """Return the embedding of the subset after applying flows."""
        return self.joint_embedding.embedding

    def get_lreval_data(self):
        lr_data = {'zk': {}}

        for key in self.subsets:
            # get the latents after application of flows.
            lr_data['zk'][key] = self.get_zk(key).cpu()

        lr_data['zk']['joint'] = self.get_joint_zk().cpu()

        return lr_data


@dataclass
class JointLatentsGfM:
    """Joint Latens for generalized f-means methods."""
    joint_embedding: JointEmbeddingFoEM
    subset_samples: Mapping[str, Tensor]
    subsets: Mapping[str, Tensor]

    def get_joint_embeddings(self):
        return self.joint_embedding.embedding

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key]

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding
        return self.subsets[subset_key]

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].cpu()

        lr_data['q0']['joint'] = self.subsets['_'.join(self.joint_embedding.mod_strs)].cpu()

        return lr_data


@dataclass
class SubsetMoFoGfM:
    """Subset of Mixture of Flow of GfM"""
    z0: Optional[Tensor] = None
    zk: Optional[Tensor] = None
    log_det_j: Optional[Tensor] = None


@dataclass
class JointLatentsMoGfM(JointLatentsGfM):
    """Joint Latents for mixture of generalized f-means methods."""

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].cpu()

        lr_data['q0']['joint'] = self.joint_embedding.embedding.cpu()

        return lr_data


@dataclass
class JointLatentsGfMoP(JointLatentsGfM):
    subsets: Mapping[str, Distr]

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].reparameterize()

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding
        return self.subsets[subset_key].mu

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].mu.cpu()

        lr_data['q0']['joint'] = self.get_joint_embeddings().cpu()

        return lr_data


@dataclass
class JointLatentsEGfM(JointLatentsGfM):
    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key]

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding
        return self.subsets[subset_key]

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].cpu()

        lr_data['q0']['joint'] = self.subsets['_'.join(self.joint_embedding.mod_strs)].cpu()

        return lr_data


@dataclass
class Joint_embeddings:
    zk: Tensor
    z0: Tensor
    mod_strs: Iterable[str]
    log_det_j: Tensor


@dataclass
class JointLatentsFoJ:
    """Joint latents of flow of joints methods"""
    joint_embedding: Joint_embeddings
    subsets: Mapping[str, Distr]

    def get_joint_embeddings(self):
        return self.joint_embedding.zk

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].reparameterize()

    def get_q0(self, subset_key: str):
        """Return the mean of q0."""
        if subset_key == 'joint':
            return self.joint_embedding.z0
        else:
            return self.subsets[subset_key].mu

    def get_joint_q0(self):
        return self.joint_embedding.z0

    def get_zk(self, subset_key: str):
        """Return the embedding of the subset after applying flows."""
        return self.joint_embedding.zk

    def get_joint_zk(self):
        """Return the embedding of the joint after applying flows."""
        return self.joint_embedding.zk

    def get_lreval_data(self):
        lr_data = {'q0': {}, 'zk': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.get_q0(key).cpu()
        lr_data['q0']['joint'] = self.get_joint_q0().cpu()

        lr_data['zk']['joint'] = self.get_joint_zk().cpu()

        return lr_data


@dataclass
class JointEmbeddingFoS:
    """Joint embedding for the flow of subsets methods."""
    embedding: Tensor
    mod_strs: Iterable[str]
    log_det_j: Tensor


@dataclass
class SubsetFoS:
    q0: Distr
    z0: Optional[Tensor] = None
    zk: Optional[Tensor] = None
    log_det_j: Optional[Tensor] = None


@dataclass
class JointLatentsFoS:
    joint_embedding: JointEmbeddingFoS
    subsets: Mapping[str, SubsetFoS]

    def get_joint_embeddings(self):
        return self.joint_embedding.embedding

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].zk

    def get_zk(self, subset_key: str):
        """Return the embedding of the subset after applying flows."""
        if subset_key == 'joint':
            return self.get_joint_zk()
        return self.subsets[subset_key].zk

    def get_q0(self, subset_key: str):
        """Return the mean of q0."""
        if subset_key == 'joint':
            return self.get_joint_q0()
        else:
            return self.subsets[subset_key].q0.mu

    def get_joint_q0(self):
        """Return the mean of q0 joint."""
        joint_mod_str = '_'.join(self.joint_embedding.mod_strs)
        return self.subsets[joint_mod_str].q0.mu

    def get_joint_zk(self):
        """Return joint zk."""
        joint_mod_str = '_'.join(self.joint_embedding.mod_strs)
        return self.subsets[joint_mod_str].zk

    def get_lreval_data(self):
        lr_data = {'q0': {}, 'zk': {}}
        for key in self.subsets:
            lr_data['q0'][key] = self.get_q0(key).cpu()

        lr_data['q0']['joint'] = self.get_joint_q0().cpu()

        for key in self.subsets:
            # get the latents after application of flows.
            lr_data['zk'][key] = self.get_zk(key).cpu()

        lr_data['zk']['joint'] = self.get_joint_zk().cpu()

        return lr_data

    # def get_lreval_data(self, data_train: dict):
    #     """Add lr values to data_train."""
    #     for key in self.subsets:
    #         data_train['q0'][key] = torch.cat((data_train['q0'][key], self.get_q0(key).cpu()), 0)
    #     joint_q0 = self.get_joint_q0().cpu()
    #     data_train['q0']['joint'] = torch.cat((data_train['q0']['joint'], joint_q0), 0)
    #
    #     for key in self.subsets:
    #         # get the latents after application of flows.
    #         data_train['zk'][key] = torch.cat((data_train['zk'][key], self.get_zk(key).cpu()), 0)
    #     joint_zk = self.get_joint_zk().cpu()
    #     data_train['zk']['joint'] = torch.cat((data_train['zk']['joint'], joint_zk), 0)
    #
    #     return data_train


@dataclass
class JointLatentsMoFoGfM(JointLatentsFoS):
    """Joint Latens for Mixture of Flow of generalized f-means methods."""
    joint_embedding: JointEmbeddingFoEM
    subsets: Mapping[str, SubsetMoFoGfM]

    def get_z0(self, subset_key: str):
        """Return the mean of q0."""
        if subset_key == 'joint':
            return self.get_joint_q0()
        else:
            return self.subsets[subset_key].z0

    def get_joint_z0(self):
        """Return joint z0."""
        return self.joint_embedding.embedding

    def get_lreval_data(self):
        lr_data = {'q0': {}, 'zk': {}}
        for key in self.subsets:
            lr_data['q0'][key] = self.get_z0(key).cpu()

        lr_data['q0']['joint'] = self.get_joint_z0().cpu()

        for key in self.subsets:
            # get the latents after application of flows.
            lr_data['zk'][key] = self.get_zk(key).cpu()

        return lr_data


@dataclass
class JointLatentsMoFoP(JointLatentsFoS):

    def get_joint_zk(self):
        """Return joint zk."""
        return self.joint_embedding.embedding

    def get_lreval_data(self):
        lr_data = {'q0': {}, 'zk': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.get_q0(key).cpu()

        for key in self.subsets:
            # get the latents after application of flows.
            lr_data['zk'][key] = self.get_zk(key).cpu()
        lr_data['zk']['joint'] = self.get_joint_zk().cpu()

        return lr_data


@dataclass
class BaseForwardResults:
    enc_mods: Mapping[str, BaseEncMod]
    joint_latents: JointLatents
    rec_mods: dict


@dataclass
class BaseBatchResults:
    total_loss: Tensor
    klds: Mapping[str, float]
    log_probs: dict
    joint_divergence: dict
    latents: Mapping[str, BaseEncMod]
    # joint_latents: Mapping[str, Tensor]


@dataclass
class BaseTestResults(BaseBatchResults):
    joint_div: float
    prd_scores: Optional[dict] = None
    lr_eval_q0: Optional[dict] = None
    lr_eval_zk: Optional[dict] = None
    gen_eval: Optional[dict] = None
    lhoods: Optional[dict] = None
    end_epoch: Optional[int] = None
    mean_epoch_time: Optional[float] = None
    experiment_duration: Optional[float] = None
    hyperopt_score: Optional[int] = None


@dataclass
class ReparamLatent:
    """Mean of latent with multiple styles."""
    content: Tensor
    style: Optional[Mapping[str, Tensor]] = None
