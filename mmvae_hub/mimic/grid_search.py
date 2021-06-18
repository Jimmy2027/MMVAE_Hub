from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.mimic.MimicTrainer import MimicTrainer
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.mimic.flags import MimicFlagsSetup
from mmvae_hub.mimic.flags import parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_spaces = {
    # 'method': ['pfom'],
    'method': ['fomfop'],
    # 'method': ['joint_elbo'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

search_spaces_1 = {
    'method': ['joint_elbo'],
    # 'method': ['moe'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False]
}

if __name__ == '__main__':

    for grid in [search_spaces_1]:
        for sp in ParameterGrid(grid):
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = MimicFlagsSetup(get_config_path(dataset='mimic',flags=flags))
            flags = flags_setup.setup(flags, additional_args=sp)
            with norby(f'Starting Experiment {flags.experiment_uid}.', f'Experiment {flags.experiment_uid} finished.'):
                mst = MimicExperiment(flags)
                mst.set_optimizer()
                trainer = MimicTrainer(mst)
                trainer.run_epochs()
