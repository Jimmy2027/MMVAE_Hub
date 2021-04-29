# -*- coding: utf-8 -*-
from pathlib import Path

import mmvae_hub
from mmvae_hub.polymnist import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser


def set_me_up(tmpdirname, method:str):
    flags = parser.parse_args([])
    config_path = Path(mmvae_hub.__file__).parent.parent / 'configs/toy_config.json'
    flags_setup = FlagsSetup(config_path)
    flags = flags_setup.setup_test(flags, tmpdirname)
    flags.method = method
    flags.method = 'planar_mixture'
    mst = PolymnistExperiment(flags)
    mst.set_optimizer()
    return mst
