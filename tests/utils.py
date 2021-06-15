# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

import mmvae_hub
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.polymnist.experiment import PolymnistExperiment


def set_me_up(tmpdirname, dataset: str, method: str, attributes: Optional = None):
    config_path = Path(mmvae_hub.__file__).parent.parent / f'configs/toy_config.json'

    if dataset == 'polymnist':
        from mmvae_hub.polymnist.flags import FlagsSetup, parser as polymnist_parser
        flags = polymnist_parser.parse_args([])
        flags_setup = FlagsSetup(config_path)
        exp = PolymnistExperiment

    elif dataset == 'mimic':
        from mmvae_hub.mimic.flags import parser as mimic_parser, MimicFlagsSetup
        flags = mimic_parser.parse_args([])
        flags_setup = MimicFlagsSetup(config_path)
        exp = MimicExperiment

    else:
        raise NotImplementedError(f'not implemented for dataset {dataset}.')

    flags = flags_setup.setup_test(flags, tmpdirname)
    flags.method = method

    if attributes:
        for k, v in attributes.items():
            setattr(flags, k, v)

    mst = exp(flags)
    mst.set_optimizer()
    return mst
