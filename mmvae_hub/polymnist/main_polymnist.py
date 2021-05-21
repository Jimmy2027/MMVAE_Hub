# -*- coding: utf-8 -*-
import os
import zipfile
from pathlib import Path

from norby.utils import maybe_norby

from mmvae_hub.base.utils.flags_utils import get_config_path
from mmvae_hub.polymnist import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser

if __name__ == '__main__':

    flags = parser.parse_args()
    flags_setup = FlagsSetup(get_config_path(flags))
    flags = flags_setup.setup(flags)

    if Path('/cluster').exists():
        polymnist_zip_path = Path('/cluster/work/vogtlab/Projects/Polymnist/PolyMNIST.zip')
        tmpdir = Path(os.getenv("TMPDIR"))
        out_dir = tmpdir / 'polymnist'
        out_dir.mkdir()

        with zipfile.ZipFile(polymnist_zip_path) as z:
            z.extractall(str(out_dir))

        flags.dir_data = out_dir

    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):
        flags = flags_setup.setup(flags)
        mst = PolymnistExperiment(flags)
        mst.set_optimizer()
        trainer = PolymnistTrainer(mst)
        trainer.run_epochs()
