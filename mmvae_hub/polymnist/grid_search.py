from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_spaces = {
    # 'method': ['mofop'],
    # 'method': ['fomop'],
    'method': ['pgfm'],
    # "eval_freq_fid": [1],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [1],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False],
}

search_spaces_1 = {
    'method': ['mofop'],
    # "dataloader_workers":[0],
    # 'method': ['moe'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [0],
    "num_mods": [1],
    "end_epoch": [500],
    "weighted_mixture": [False]
}

if __name__ == '__main__':

    for grid in [search_spaces_1]:
        for sp in ParameterGrid(grid):
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = FlagsSetup(get_config_path(dataset='polymnist', flags=flags))
            flags = flags_setup.setup(flags, additional_args=sp)
            with norby(f'Starting Experiment {flags.experiment_uid}.', f'Experiment {flags.experiment_uid} finished.'):
                mst = PolymnistExperiment(flags)
                mst.set_optimizer()
                trainer = PolymnistTrainer(mst)
                trainer.run_epochs()
