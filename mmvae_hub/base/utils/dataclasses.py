# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


@dataclass
class BaseLatents:
    enc_mods: dict
    joint: dict


@dataclass
class BaseEncMods:
    latents_class: dict
    latents_style: dict


@dataclass
class EncModsPlanarFlow(BaseEncMods):
    flow_params: dict
    z0: Tensor
    zk: Tensor
    log_det_j: Tensor


@dataclass
class BaseForwardResults:
    enc_mods: dict
    joint_latents: dict
    rec_mods: dict


@dataclass
class BaseDivergences:
    joint_div: float
    mods_div: Mapping[str, float]
