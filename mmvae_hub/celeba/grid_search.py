from norby.utils import norby
from sklearn.model_selection import ParameterGrid

from mmvae_hub.celeba.CelebaTrainer import CelebaTrainer
from mmvae_hub.celeba.experiment import CelebaExperiment
from mmvae_hub.celeba.flags import CelebaFlagsSetup, parser
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
    'method': ['iwmogfm_amortized'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "dataloader_workers": [8],
    "max_beta": [1.0e-04, 1.0e-02, 0.5],
    "beta_start_epoch": [30],
    "beta_warmup": [0],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [100],
    "calc_nll": [False],
    "K": [5],
    "eval_freq": [20],
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
    'method': ['mopoe'],
    'max_beta': [2.],
    "beta_warmup": [0],
    'class_dim': [1280],
    "initial_learning_rate": [0.0005],
    "end_epoch": [100],
    "eval_freq": [10],
    # "K":[5]
    # "factorized_representation": [True],
    # "beta": [5.],
    # "beta_style": [2.0],
    # "beta_content": [1.0],
    # "beta_m1_style": [1.0],
    # "beta_m2_style": [5.0],
    # "style_img_dim": [32],
    # "style_text_dim": [32],

}

if __name__ == '__main__':
    dataset = 'celeba'

    for grid in [sp_joint_elbo_article]:
        for sp in ParameterGrid(grid):
            # for _ in [1]:
            flags = parser.parse_args()
            flags_setup = CelebaFlagsSetup(get_config_path(dataset='celeba', flags=flags))
            flags = flags_setup.setup(flags, additional_args={**sp, 'dataset': dataset})

            with norby(f'Starting Experiment {flags.experiment_uid}.', f'Experiment {flags.experiment_uid} finished.'):
                mst = CelebaExperiment(flags)
                mst.set_optimizer()
                trainer = CelebaTrainer(mst)
                trainer.run_epochs()
