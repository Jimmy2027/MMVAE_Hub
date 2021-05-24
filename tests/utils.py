# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

import mmvae_hub
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser


def set_me_up(tmpdirname, method: str, attributes: Optional = None):
    flags = parser.parse_args([])
    config_path = Path(mmvae_hub.__file__).parent.parent / 'configs/toy_config.json'
    flags_setup = FlagsSetup(config_path)
    flags = flags_setup.setup_test(flags, tmpdirname)
    flags.method = method

    if attributes:
        for k, v in attributes.items():
            setattr(flags, k, v)

    # flags.use_db = True
    mst = PolymnistExperiment(flags)
    mst.set_optimizer()
    return mst
