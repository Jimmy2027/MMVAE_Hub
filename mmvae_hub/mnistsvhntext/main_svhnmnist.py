from norby.utils import maybe_norby

from mmvae_hub import log
from mmvae_hub.leomed_utils.boilerplate import compress_experiment_run_dir
from mmvae_hub.mnistsvhntext.experiment import MNISTSVHNText
from mmvae_hub.mnistsvhntext.flags import mnistsvhntextFlagsSetup, parser
from mmvae_hub.mnistsvhntext.mnistsvhntextTrainer import mnistsvhnTrainer
from mmvae_hub.utils.setup.flags_utils import get_config_path

DATASET = 'mnistsvhntext'

if __name__ == '__main__':

    flags = parser.parse_args()
    flags_setup = mnistsvhntextFlagsSetup(get_config_path(dataset=DATASET, flags=flags))
    flags = flags_setup.setup(flags, additional_args={'dataset': DATASET})

    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):

        mst = MNISTSVHNText(flags)

        mst.set_optimizer()
        trainer = mnistsvhnTrainer(mst)
        trainer.run_epochs()

    log.info('Done.')
    # move zipped experiment_dir_run in TMPDIR to experiment_dir
    if flags.leomed:
        compress_experiment_run_dir(flags)
