from norby.utils import maybe_norby

from mmvae_hub.VQVAE.VQTrainer import VQTrainer
from mmvae_hub.VQVAE.vq_mimic_experiment import VQmimicExperiment
from mmvae_hub.leomed_utils.boilerplate import compress_experiment_run_dir
from mmvae_hub.mimic.flags import parser, MimicFlagsSetup
from mmvae_hub.utils.setup.flags_utils import get_config_path

DATASET = 'mimic'

if __name__ == '__main__':

    flags = parser.parse_args()

    flags.dataset = DATASET
    flags_setup = MimicFlagsSetup(get_config_path(dataset=DATASET, flags=flags))
    flags = flags_setup.setup(flags, additional_args={'dataset': DATASET, 'method': 'vqmogfm'})

    # Args
    # flags.mods = 'T'
    flags.batch_size = 128
    # flags.end_epoch = 150

    flags.num_hiddens = 128
    flags.num_residual_hiddens = 32
    flags.num_residual_layers = 2

    flags.embedding_dim = 64
    flags.num_embeddings = 512

    flags.commitment_cost = 0.25

    flags.decay = 0.99

    flags.eval_freq = 20
    flags.eval_lr = False
    flags.eval_freq_fid = 10000
    flags.calc_prd = False
    flags.use_clf = True

    with maybe_norby(flags.norby, f'Starting Experiment {flags.experiment_uid}.',
                     f'Experiment {flags.experiment_uid} finished.'):
        exp = VQmimicExperiment(flags)
        exp.set_optimizer()
        trainer = VQTrainer(exp)
        trainer.run_epochs()

    # move zipped experiment_dir_run in TMPDIR to experiment_dir
    if flags.leomed:
        compress_experiment_run_dir(flags)
