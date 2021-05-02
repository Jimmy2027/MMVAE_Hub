# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Mapping, Optional

from torch import Tensor


@dataclass
class BaseLatents:
    enc_mods: dict
    joint: dict


@dataclass
class BaseDivergences:
    joint_div: float
    mods_div: Mapping[str, Tensor]


@dataclass()
class Distr:
    mu: Tensor
    logvar: Tensor


@dataclass
class PlanarFlowParams:
    u: Tensor
    w: Tensor
    b: Tensor


@dataclass
class EncModPlanarMixture:
    latents_class: Distr
    flow_params: PlanarFlowParams
    z0: Optional[Tensor] = None
    zk: Optional[Tensor] = None
    log_det_j: Optional[Tensor] = None
    latents_style: Optional[Distr] = None


@dataclass
class BaseEncMod:
    # latents have shape [batch_size, class_dim]
    latents_class: Distr
    latents_style: Optional[Distr] = None


@dataclass
class JointLatents:
    mus: Tensor
    logvars: Tensor
    joint_distr: Distr
    subsets: Mapping[str, Distr]


@dataclass
class BaseForwardResults:
    enc_mods: Mapping[str, BaseEncMod]
    joint_latents: JointLatents
    rec_mods: dict


@dataclass
class BaseTestResults:
    joint_div: float
    prd_scores: Optional[dict] = None
    lr_eval: Optional[dict] = None
    gen_eval: Optional[dict] = None
    lhoods: Optional[dict] = None
    end_epoch: Optional[int] = None
    mean_epoch_time: Optional[float] = None
    experiment_duration: Optional[float] = None


@dataclass
class ReparamLatent:
    content: Tensor
    style: Optional[Mapping[str, Tensor]] = None
