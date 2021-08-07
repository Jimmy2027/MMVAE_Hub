from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.polymnist.flags import FlagsSetup, parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

search_spaces = {
    'method': ['joint_elbo'],
    'beta': [1.4810022901262143],
    'class_dim': [640],
    "num_mods": [3],
    "initial_learning_rate": [0.0006212184464462084],
    "end_epoch": [150],
}

search_spaces_1 = {
    'method': ['mopgfm'],
    "initial_learning_rate": [0.0009253348001968961],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [1.5142062143401498],
    "beta_warmup": [50],
    "num_gfm_flows": [3],
    "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [150],
}

sp_joint_elbo_article = {
    'n_gpus': [1],
    'method': ['joint_elbo'],
    'max_beta': [2.5],
    "beta_warmup": [0],
    'class_dim': [512],
    "num_mods": [5],
    "initial_learning_rate": [0.001],
    "end_epoch": [300],
}

search_space_sylvester = {
    'method': ['planar_vae'],
    'max_beta': [2.5],
    'class_dim': [640],
    "num_mods": [1],
    "initial_learning_rate": [0.0001],
    "end_epoch": [50],
    "calc_nll": [False]
}

if __name__ == '__main__':

    for grid in [search_space_sylvester]:
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
