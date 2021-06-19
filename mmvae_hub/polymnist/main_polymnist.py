from norby.utils import maybe_norby

from mmvae_hub import log
from mmvae_hub.leomed_utils.boilerplate import compress_experiment_run_dir
from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

DATASET = 'polymnist'

if __name__ == '__main__':

    flags = parser.parse_args()
    flags_setup = FlagsSetup(get_config_path(dataset=DATASET, flags=flags))
    flags = flags_setup.setup(flags, additional_args={'dataset': DATASET})

    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):
        mst = PolymnistExperiment(flags)
        mst.set_optimizer()
        trainer = PolymnistTrainer(mst)
        trainer.run_epochs()

    log.info('Done.')
    # move zipped experiment_dir_run in TMPDIR to experiment_dir
    if flags.leomed:
        compress_experiment_run_dir(flags)
