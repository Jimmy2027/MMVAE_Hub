import shutil
from pathlib import Path

from norby.utils import maybe_norby

from mmvae_hub import log
from mmvae_hub.base.utils.flags_utils import get_config_path
from mmvae_hub.base.utils.utils import json2dict
from mmvae_hub.polymnist import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser

if __name__ == '__main__':

    flags = parser.parse_args()
    flags_setup = FlagsSetup(get_config_path(flags))
    flags = flags_setup.setup(flags)

    log.info(f'dir_experiment_run: {flags.dir_experiment_run}')
    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):
        mst = PolymnistExperiment(flags)
        mst.set_optimizer()
        trainer = PolymnistTrainer(mst)
        trainer.run_epochs()

    # move zipped experiment_dir_run in TMPDIR to experiment_dir
    if flags.leomed:
        dir_experiment = Path(json2dict(get_config_path())['dir_experiment'])
        dir_experiment.mkdir(exist_ok=True)

        # zip dir_experiment_run
        log.info(f'zipping {flags.dir_experiment_run} '
                 f'to {(Path(dir_experiment) / flags.experiment_uid).with_suffix("zip")}.')
        dir_experiment_zipped = (Path(dir_experiment) / flags.experiment_uid).with_suffix('zip')
        shutil.make_archive(dir_experiment_zipped, 'zip',
                            flags.dir_experiment_run)

        assert dir_experiment_zipped.exists(), f'{dir_experiment_zipped} does not exist. Zipping of dir_experiment_run failed.'

    log.info('Done.')
