from norby.utils import maybe_norby

from mmvae_hub.leomed_utils.boilerplate import compress_experiment_run_dir
from mmvae_hub.mimic.MimicTrainer import MimicTrainer
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.mimic.flags import parser, MimicFlagsSetup
from mmvae_hub.utils.setup.flags_utils import get_config_path

if __name__ == '__main__':

    flags = parser.parse_args()
    flags.dataset = 'mimic'
    flags_setup = MimicFlagsSetup(get_config_path(flags= flags))
    flags = flags_setup.setup(flags)

    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):
        exp = MimicExperiment(flags)
        exp.set_optimizer()
        trainer = MimicTrainer(exp)
        trainer.run_epochs()

    # move zipped experiment_dir_run in TMPDIR to experiment_dir
    if flags.leomed:
        compress_experiment_run_dir(flags)
