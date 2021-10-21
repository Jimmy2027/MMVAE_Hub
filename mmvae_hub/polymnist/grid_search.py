from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_spaces = {
    'method': ['iwmopgfm_'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [2.],
    "beta_warmup": [50],
    "num_mods": [3],
    "end_epoch": [150],
    "calc_nll": [False],
    "K": [1],
    "eval_freq": [50],
    "num_gfm_flows": [2],
    'gpu_mem': [10000],
    "qz_x": ['normal']
}

search_spaces_1 = {
    'method': ['iwmogfm'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [1.],
    "beta_warmup": [50],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [150],
    "calc_nll": [False],
    "K": [5],
    "eval_freq": [50],
}

search_spaces_2 = {
    'method': ['iwmogfm2_'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "dataloader_workers": [12],
    "max_beta": [0.001],
    "beta_start_epoch": [30.],
    "beta_warmup": [50],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [500],
    "calc_nll": [False],
    "K": [2],
    "eval_freq": [100],
}

search_spaces_amortized = {
    'method': ['iwmogfm_amortized'],
    "initial_learning_rate": [0.0005],
    'class_dim': [1280],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [0.],
    "beta_start_epoch": [0.],
    "beta_warmup": [50],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [2],
    "end_epoch": [500],
    "calc_nll": [False],
    "K": [1],
    "eval_freq": [100],
}

search_spaces_3 = {
    'method': ['iwmogfm4'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [1.],
    "beta_start_epoch": [0],
    "beta_warmup": [0],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [100],
    "calc_nll": [False],
    "K": [5],
    "eval_freq": [20],
}

sp_joint_elbo_article = {
    'n_gpus': [1],
    'method': ['poe'],
    'max_beta': [2.5],
    "beta_warmup": [0],
    'class_dim': [1280],
    "num_mods": [3],
    "initial_learning_rate": [0.001],
    "end_epoch": [1],
    "eval_freq": [1],
    "calc_nll": [False],
}
search_space_sylvester = {
    'method': ['mopoe'],
    'max_beta': [1.],
    'class_dim': [64],
    "num_mods": [1],
    "num_flows": [5],
    "initial_learning_rate": [0.0005],
    "end_epoch": [1],
    "eval_freq": [1],
    "calc_nll": [False],
}

if __name__ == '__main__':

    for grid in [search_spaces_amortized]:
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
