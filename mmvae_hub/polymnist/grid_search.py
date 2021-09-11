from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_spaces = {
    'method': ['mopgfm'],
    "initial_learning_rate": [0.0009253348001968961],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [1.5142062143401498],
    "beta_warmup": [50],
    "num_gfm_flows": [3],
    "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [100],
}

search_spaces_1 = {
    'method': ['iwmogfm'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [2.0],
    "beta_warmup": [50],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [150],
    "calc_nll": [False],
    "K": [10],
    "eval_freq": [1],
}

search_spaces_2 = {
    'method': ['iwmopgfm'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [2.0],
    "beta_warmup": [50],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [150],
    "calc_nll": [False],
    "K": [10],
    "eval_freq": [10],
}

sp_joint_elbo_article = {
    'n_gpus': [1],
    'method': ['joint_elbo'],
    'max_beta': [2.5],
    "beta_warmup": [0],
    'class_dim': [512],
    "num_mods": [3],
    "initial_learning_rate": [0.001],
    "end_epoch": [1],
}

search_space_sylvester = {
    'method': ['planar_vae'],
    'max_beta': [1.],
    'class_dim': [64],
    "num_mods": [1],
    "num_flows": [5],
    "initial_learning_rate": [0.0005],
    "end_epoch": [50],
    "calc_nll": [False]
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
