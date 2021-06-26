# -*- coding: utf-8 -*-

import typing

import numpy as np

from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.FlowVaes import FlowOfEncModsVAE, FlowOfSubsetsVAE, FoMFoP, FoMoP, GfMVAE
from mmvae_hub.utils.Dataclasses import *


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Taken from https://github.com/pytorch/examples/blob/a3f28a26851867b314f4471ec6ca1c2c048217f1/imagenet/main.py#L363
    """

    def __init__(self, name: str, fmt: str = ':f', precision=None):
        self.name = name
        self.precision = precision
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def get_average(self):
        if self.precision:
            return np.round(self.avg, self.precision)
        return self.avg


class AverageMeterNestedDict:
    """
    Computes and stores the average and current value
    Inspired by https://github.com/pytorch/examples/blob/a3f28a26851867b314f4471ec6ca1c2c048217f1/imagenet/main.py#L363
    """

    def __init__(self, name: str, structure: typing.Mapping[str, typing.Mapping[str, typing.Iterable[None]]]):
        self.name = name
        self.structure = structure
        self.vals: typing.Mapping[str, typing.Mapping[str, typing.Iterable[typing.Optional[float]]]] = structure

    def update(self, val: typing.Mapping[str, typing.Mapping[str, typing.Iterable[float]]]) -> None:
        for k1 in self.structure:
            for k2 in self.structure:
                self.vals[k1][k2].append(val[k1][k2])

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, float]]:
        d = {}
        for k1 in self.structure:
            for k2 in self.structure:
                d[k1][k2] = np.mean(self.vals[k1][k2])

        return d


class AverageMeterDict:
    """
    Computes and stores the average and current value
    Inspired by https://github.com/pytorch/examples/blob/a3f28a26851867b314f4471ec6ca1c2c048217f1/imagenet/main.py#L363
    """

    def __init__(self, name: str):
        self.name = name
        self.vals: typing.Optional[typing.Mapping[str, typing.Iterable[float]]] = None

    def update(self, val: typing.Mapping[str, typing.Iterable[float]]) -> None:
        if not self.vals:
            self.vals = {k: [] for k in val}
        for key in val:
            self.vals[key].append(val[key])

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, float]]:
        return {key: np.mean(self.vals[key]) for key in self.vals}


class AverageMeterLatents(AverageMeterDict):
    def __init__(self, name: str, factorized_representation: bool):
        super().__init__(name=name)
        self.factorized_representation = factorized_representation

    def update(self, val: typing.Mapping[str, BaseEncMod]):
        if not self.vals:
            lvl2_keys = ['latents_class', 'latents_style'] if self.factorized_representation else ['latents_class']
            self.vals = {l1: {l2: {'mu': [], 'logvar': []} for l2 in lvl2_keys} for l1 in [k for k in val]}

        for mod_string, enc_mods in val.items():
            for key in self.vals[mod_string]:
                self.vals[mod_string][key]['mu'].append(val[mod_string].__dict__[key].mu.mean().item())
                self.vals[mod_string][key]['logvar'].append(val[mod_string].__dict__[key].logvar.mean().item())

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, typing.Tuple[float, float]]]:
        return {mod_str: {k: {'mu': np.mean(v['mu']), 'logvar': np.mean(v['logvar'])} for k, v in enc_mods.items()} for
                mod_str, enc_mods in self.vals.items()}


class AverageMeterJointLatents(AverageMeterDict):
    def __init__(self, name: str, factorized_representation: bool, model: BaseMMVAE):
        super().__init__(name=name)
        self.factorized_representation = factorized_representation
        self.model = model
        self.vals = None

    def update(self, val: typing.Union[JointLatents, JointLatentsFoEM]):
        """Need to differentiate between 4 types of methods.
        - no flows
        - methods where the flow is applied on the encoding of each modality
        - methods where the flow is applied on each subset
        - methods where the flow is applied on the joint distribution

        """
        if not self.vals:
            init_val = [] if isinstance(self.model, (FlowOfEncModsVAE, FlowOfSubsetsVAE, FoMFoP)) else {'mu': [],
                                                                                                        'logvar': []}
            self.vals = {k: init_val for k in list(val.subsets)}
            self.vals['joint'] = [] if isinstance(self.model, FoMoP) else init_val

        if isinstance(self.model, FlowOfEncModsVAE):
            for subset_key, subset in val.subsets.items():
                self.vals[subset_key].append(subset.mean().item())
            self.vals['joint'].append(val.joint_embedding.embedding.mean().item())

        elif isinstance(self.model, (FlowOfSubsetsVAE, FoMFoP)):
            for subset_key, subset in val.subsets.items():
                self.vals[subset_key].append(subset.zk.mean().item())
            self.vals['joint'].append(val.joint_embedding.embedding.mean().item())

        elif isinstance(self.model, FoMoP):
            for subset_key, subset in val.subsets.items():
                self.vals[subset_key]['mu'].append(subset.mu.mean().item())
                self.vals[subset_key]['logvar'].append(subset.logvar.mean().item())
            self.vals['joint'].append(val.joint_embedding.zk.mean().item())

        elif isinstance(self.model, GfMVAE):
            for subset_key, subset in val.subsets.items():
                self.vals[subset_key]['mu'].append(subset.mu.mean().item())
                self.vals[subset_key]['logvar'].append(subset.logvar.mean().item())
            self.vals['joint'].append(val.joint_embedding.embedding.mean().item())
        else:
            for subset_key, subset in val.subsets.items():
                self.vals[subset_key]['mu'].append(subset.mu.mean().item())
                self.vals[subset_key]['logvar'].append(subset.logvar.mean().item())
            self.vals['joint']['mu'].append(val.joint_distr.mu.mean().item())
            self.vals['joint']['logvar'].append(val.joint_distr.logvar.mean().item())

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, typing.Tuple[float, float]]]:
        if isinstance(self.model, (FlowOfEncModsVAE, FlowOfSubsetsVAE, FoMFoP)):
            return {k: np.mean(v) for k, v in self.vals.items()}

        elif isinstance(self.model, FoMoP):
            results = {}
            for k, v in self.vals.items():
                if k == 'joint':
                    results[k] = np.mean(v)
                else:
                    results[k] = {'mu': np.mean(v['mu']), 'logvar': np.mean(v['logvar'])}

        else:
            return {mod_str: {'mu': np.mean(v['mu']), 'logvar': np.mean(v['logvar'])} for mod_str, v in
                    self.vals.items()}
