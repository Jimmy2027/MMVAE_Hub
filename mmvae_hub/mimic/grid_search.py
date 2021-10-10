from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.mimic.MimicTrainer import MimicTrainer
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.mimic.flags import MimicFlagsSetup
from mmvae_hub.mimic.flags import parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

sp_mopoe_mimic = {
    'method': ['mopoe'],
    'beta': [2.],
    'class_dim': [1280],
    # "num_mods": [1],
    # "num_flows": [5],
    "initial_learning_rate": [5e-04],
    "end_epoch": [1],
    "feature_extractor_img": ['resnet'],
    # "coupling_dim": [512],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

search_spaces_1 = {
    'mods': ['F_L_T'],
    # 'method': ['moe'],
    'method': ['mopoe'],
    "initial_learning_rate": [0.0009253348001968961],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [1.5142062143401498],
    "beta_warmup": [50],
    "num_gfm_flows": [3],
    "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [250],
}

if __name__ == '__main__':

    for grid in [sp_mopoe_mimic]:
        for sp in ParameterGrid(grid):
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = MimicFlagsSetup(get_config_path(dataset='mimic', flags=flags))
            flags = flags_setup.setup(flags, additional_args=sp)
            with norby(f'Starting Experiment {flags.experiment_uid}.', f'Experiment {flags.experiment_uid} finished.'):
                mst = MimicExperiment(flags)
                mst.set_optimizer()
                trainer = MimicTrainer(mst)
                trainer.run_epochs()
