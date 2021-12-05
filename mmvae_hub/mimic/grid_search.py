from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.mimic.MimicTrainer import MimicTrainer
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.mimic.flags import MimicFlagsSetup
from mmvae_hub.mimic.flags import parser
from mmvae_hub.utils.setup.flags_utils import get_config_path

sp_mopoe_mimic = {
    'method': ['mopgfm'],
    'mods': ['F'],
    'beta': [2.],
    'class_dim': [640],
    # "num_mods": [1],
    # "num_flows": [5],
    "initial_learning_rate": [5e-04],
    "end_epoch": [150],
    "eval_freq": [150],
    "beta_warmup": [0],
    # "batch_size":[64]
    # "coupling_dim": [512],
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

search_spaces_amortized = {
    'method': ['iwmogfm_amortized'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "dataloader_workers": [16],
    "max_beta": [0.],
    "beta_start_epoch": [0.],
    "beta_warmup": [50],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "coupling_dim": [64],
    "num_gfm_flows": [3],
    "nbr_coupling_block_layers": [8],
    "end_epoch": [150],
    "calc_nll": [False],
    "K": [1],
    "eval_freq": [150],
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
