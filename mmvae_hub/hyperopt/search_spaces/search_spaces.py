pfom_optuna_sp = {
    'n_gpus': [1],
    # the end_epoch here is used to evaluate the time needed for the run.
    # The actual end_epoch parameter is set by the HyperoptTrainer.
    "end_epoch": [1000],
    'method': ['pfom'],
    "num_mods": [3],
    "optuna": [True],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

sp_sylvester = {
    'n_gpus': [1],
    'method': ['sylvester_vae_noflow'],
    'min_beta': [0],
    'max_beta': [1.],
    "beta_warmup": [50],
    'class_dim': [640],
    "num_mods": [3],
    "num_gfm_flows": [3],
    "initial_learning_rate": [0.0005],
    "end_epoch": [100],
    "coupling_dim": [32],
}

sp_mopoe = {
    'n_gpus': [1],
    'method': ['mopoe'],
    'max_beta': [1.],
    'class_dim': [1280],
    "beta_warmup": [50],
    "num_mods": [3],
    "initial_learning_rate": [0.0005],
    "end_epoch": [300],
}

sp_mopoe_mnistsvhntext = {
    'n_gpus': [1],
    'gpu_mem': [30000],
    'method': ['mopoe'],
    'max_beta': [5.],
    'class_dim': [20],
    "beta_warmup": [0],
    "initial_learning_rate": [0.001],
    "end_epoch": [150],
    "data_multiplications": [10]
}
sp_mofop_mnistsvhntext = {
    'n_gpus': [1],
    'gpu_mem': [15000],
    'method': ['mofop', 'mopgfm'],
    'max_beta': [1.3],
    'class_dim': [512],
    "beta_warmup": [30],
    "initial_learning_rate": [0.0005],
    "nbr_coupling_block_layers": [8],
    "coupling_dim": [64],
    "end_epoch": [100],
    "num_gfm_flows": [3],
    "data_multiplications": [10],
    "calc_nll": [False]
}

sp_mopgfm = {
    'n_gpus': [1],
    'method': ['mopgfm'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [2.],
    "beta_warmup": [50],
    "num_gfm_flows": [1],
    "coupling_dim": [64],
    "nbr_coupling_block_layers": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "eval_freq": [10]
}

iwmoe = {
    'method': ['iwmoe'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "max_beta": [2.5],
    "beta_warmup": [0],
    # "num_gfm_flows": [3],
    # "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [300],
    "calc_nll": [False]
}

iwmopoe = {
    'method': ['iwpoe', 'iwpoe', 'iwpoe'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "max_beta": [2.],
    "beta_warmup": [0],
    "K": [3, 5],
    "num_mods": [3],
    "end_epoch": [500],
    "calc_nll": [False],
    'gpu_mem': [10000],
    "eval_freq": [100],
    "num_gfm_flows": [3],
    "coupling_dim": [64],
    "nbr_coupling_block_layers": [8],
}

sp_iwmopgfm = {
    'method': ['iwmopgfm'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "max_beta": [2.],
    "beta_warmup": [50],
    "K": [3, 5],
    "num_mods": [3],
    "end_epoch": [500],
    "calc_nll": [False],
    'gpu_mem': [15000],
    "eval_freq": [100],
    "num_gfm_flows": [3],
    "coupling_dim": [64],
    "nbr_coupling_block_layers": [8],
}

iwmogfm2 = {
    # 'method': ['iwmogfm_amortized'],
    'method': ['iwmogfm2_'],
    "initial_learning_rate": [0.0005],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [0.01],
    "beta_warmup": [100],
    "beta_start_epoch": [30],
    "num_mods": [3],
    "end_epoch": [500],
    "calc_nll": [False],
    "K": [2],
    "eval_freq": [100],
    'gpu_mem': [10000],
    "num_gfm_flows": [3],
    "coupling_dim": [64],
}

sp_bmogfm = {
    'n_gpus': [1],
    'method': ['bmogfm'],
    'max_beta': [1],
    'min_beta': [0],
    "beta_warmup": [50],
    'class_dim': [256],
    # "num_mods": [3],
    "num_flows": [5],
    "eval_freq": [100],
    # "initial_learning_rate": [9e-05],
    "end_epoch": [100],
    "coupling_dim": [512],
}

sp_gfm = {
    'n_gpus': [1],
    'method': ['gfm'],
    'max_beta': [1],
    'min_beta': [0],
    "beta_warmup": [50],
    'class_dim': [256],
    # "num_mods": [3],
    "num_flows": [5],
    # "initial_learning_rate": [9e-05],
    "end_epoch": [150],
    "coupling_dim": [512],
}

sp_mopoe_mimic = {
    'n_gpus': [1],
    'method': ['mopoe'],
    'beta': [1.],
    'class_dim': [512],
    "beta_warmup": [150],
    # "num_mods": [1],
    # "num_flows": [5],
    "initial_learning_rate": [0.0005],
    "end_epoch": [150],
    "eval_freq": [50],
    'gpu_mem': [15000],
    # "coupling_dim": [512],
}

mopoe_celeba = {
    'n_gpus': [1],
    'method': ['mopoe', 'mopoe', 'mopoe'],
    'max_beta': [2.],
    "beta_warmup": [0],
    'class_dim': [512],
    "initial_learning_rate": [0.0005],
    "end_epoch": [100],
    "eval_freq": [50],
    "use_clf": [True],
    "batch_size": [128]
}

mopgfm_celeba = {
    'n_gpus': [1],
    'method': ['mofop', 'mofop', 'mofop', 'mofop'],
    'max_beta': [2.],
    "beta_warmup": [50],
    'class_dim': [512],
    "initial_learning_rate": [0.0005],
    "end_epoch": [150],
    "eval_freq": [50],
    "use_clf": [True],
    "batch_size": [128],
    "coupling_dim": [64],
    "nbr_coupling_block_layers": [8],
    "num_gfm_flows": [15],
}

mopgfm_mimic = {
    'method': ['mopgfm'],
    "initial_learning_rate": [0.0005],
    # 'class_dim': [640, 1280],
    'class_dim': [64],
    "min_beta": [0],
    "dataloader_workers": [16],
    # "max_beta": [1.0, 2.5],
    "max_beta": [2.],
    "beta_warmup": [30],
    "coupling_dim": [64],
    "nbr_coupling_block_layers": [8],
    "end_epoch": [200],
    "calc_nll": [False],
    "eval_freq": [100],
    "num_gfm_flows": [1],
    'gpu_mem': [30000],

}

iwmogfm_mimic = {
    'method': ['iwmogfm'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "max_beta": [0],
    "beta_warmup": [0],
    "end_epoch": [100],
    "calc_nll": [False],
    "K": [2],
    "eval_freq": [50],
    "num_gfm_flows": [1],
    'gpu_mem': [10000],
}

iwmopgfm_mimic = {
    'n_gpus': [1],
    'method': ['iwmopgfm'],
    "K": [3],
    'max_beta': [1.],
    'class_dim': [256],
    "initial_learning_rate": [0.0005],
    # "num_mods": [3],
    "num_flows": [3],
    "beta_warmup": [30],
    "coupling_dim": [64],
    "num_gfm_flows": [3],
    "nbr_coupling_block_layers": [8],
    "end_epoch": [150],
    "use_clf": [False],
    'gpu_mem': [30000],
}

flow_mimic = {
    'n_gpus': [1],
    # 'method': ['mopgfm', 'mofop'],
    'method': ['mofop', 'mofop', 'mofop'],
    'max_beta': [1.],
    'class_dim': [512],
    "initial_learning_rate": [0.0005],
    # "num_mods": [3],
    "num_flows": [3],
    "beta_warmup": [100],
    "coupling_dim": [64],
    "num_gfm_flows": [3],
    "nbr_coupling_block_layers": [8],
    "end_epoch": [150],
    "use_clf": [False],
    'gpu_mem': [30000],
}

amortized_mimic = {
    'method': ['mogfm_amortized'],
    "initial_learning_rate": [0.0005],
    'class_dim': [512],
    "min_beta": [0],
    "max_beta": [0.],
    "beta_start_epoch": [0.],
    "beta_warmup": [0],
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


sp_mofop = {
    'n_gpus': [1],
    'method': ['mofop'],
    'max_beta': [1.],
    "beta_warmup": [50],
    'class_dim': [512],
    "num_mods": [3],
    "initial_learning_rate": [0.0005],
    "nbr_coupling_block_layers": [4],
    "end_epoch": [300],
    "num_flows": [4, 10],
    "calc_nll": [False]
}

sp_mogfm = {
    'method': ['mogfm'],
    "initial_learning_rate": [0.00084902],
    'class_dim': [1280],
    "min_beta": [0],
    "max_beta": [1.53570792849623],
    "beta_warmup": [100],
    "num_gfm_flows": [1],
    "coupling_dim": [256],
    "num_mods": [3],
    "batch_size": [128],
    "end_epoch": [500],
    'gpu_mem': [30000],
    "calc_nll": [False]
}

sp_mogfm_ = {
    'method': ['mogfm'],
    "initial_learning_rate": [0.0009253348001968961],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [1.5142062143401498],
    "beta_warmup": [50],
    "num_gfm_flows": [3],
    "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [300],
    'gpu_mem': [30000],
    "calc_nll": [False]
}

sp_joint_elbo_article = {
    'n_gpus': [1],
    'method': ['mopgfm'],
    'max_beta': [1.],
    "beta_warmup": [0],
    'class_dim': [512],
    # "num_mods": [3],
    "initial_learning_rate": [0.001],
    "end_epoch": [500],
    'gpu_mem': [12000]
}

poe_args = {
    'method': ['poe'],
    "initial_learning_rate": [0.0005],
    'class_dim': [1280],
    "min_beta": [0],
    "max_beta": [2.],
    "beta_warmup": [0],
    "num_mods": [1, 2, 3, 4, 5],
    "end_epoch": [500],
    "eval_freq": [100],
}
