from norby.utils import maybe_norby

from mmvae_hub import log
from mmvae_hub.celeba.CelebaTrainer import CelebaTrainer
from mmvae_hub.celeba.experiment import CelebaExperiment
from mmvae_hub.celeba.flags import parser, CelebaFlagsSetup
from mmvae_hub.leomed_utils.boilerplate import compress_experiment_run_dir

from mmvae_hub.utils.setup.flags_utils import get_config_path

DATASET = 'celeba'

if __name__ == '__main__':

    flags = parser.parse_args()
    flags_setup = CelebaFlagsSetup(get_config_path(dataset=DATASET, flags=flags))
    flags = flags_setup.setup(flags, additional_args={'dataset': DATASET})

    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):
        mst = CelebaExperiment(flags)
        mst.set_optimizer()
        trainer = CelebaTrainer(mst)
        trainer.run_epochs()

    log.info('Done.')
    # move zipped experiment_dir_run in TMPDIR to experiment_dir
    if flags.leomed:
        compress_experiment_run_dir(flags)
