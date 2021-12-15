from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.VQVAE.VQTrainer import VQTrainer
from mmvae_hub.VQVAE.vq_mimic_experiment import VQmimicExperiment
from mmvae_hub.mimic.flags import MimicFlagsSetup
from mmvae_hub.mimic.flags import parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_space = {
    'method': ['vqmoam'],
    'mods': ['F_L'],
    # 'mods': ['T'],
    'class_dim': [512],
    "end_epoch": [2],
    "initial_learning_rate": [1e-4],
    # "num_gfm_flows": [3, 5]

}

if __name__ == '__main__':

    for grid in [search_space]:
        for sp in ParameterGrid(grid):
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = MimicFlagsSetup(get_config_path(dataset='mimic', flags=flags))
            flags = flags_setup.setup(flags, additional_args=sp)
            with norby(f'Starting Experiment {flags.experiment_uid}.', f'Experiment {flags.experiment_uid} finished.'):
                exp = VQmimicExperiment(flags)
                exp.set_optimizer()
                trainer = VQTrainer(exp)
                trainer.run_epochs()
