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

sp_pgfm = {
    'n_gpus': [1],
    'method': ['pgfm'],
    'min_beta': [0],
    'max_beta': [1.7520493158062553],
    "beta_warmup": [50],
    'class_dim': [640],
    # "num_mods": [3],
    "num_gfm_flows": [3],
    "initial_learning_rate": [7.928188645921211e-05],
    "end_epoch": [1000],
    "coupling_dim": [32],
}

sp_mopgfm = {
    'n_gpus': [1],
    'method': ['mopgfm'],
    "initial_learning_rate": [0.0009253348001968961],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [1.5142062143401498],
    "beta_warmup": [50],
    "num_gfm_flows": [3],
    "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [500],
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

sp_joint_elbo = {
    'n_gpus': [1],
    'method': ['joint_elbo'],
    'max_beta': [1.4810022901262143],
    'class_dim': [640],
    "num_mods": [3],
    "initial_learning_rate": [0.0006212184464462084],
    "end_epoch": [500],
}

sp_joint_elbo_mimic = {
    'n_gpus': [1],
    'method': ['joint_elbo'],
    'beta': [1],
    'class_dim': [512],
    # "num_mods": [1],
    # "num_flows": [5],
    "initial_learning_rate": [5e-04],
    "end_epoch": [150],
    # "coupling_dim": [512],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

sp_pgfm_mimic = {
    'n_gpus': [1],
    'method': ['pgfm'],
    'beta': [1],
    'class_dim': [512],
    # "num_mods": [3],
    "num_flows": [1],
    # "initial_learning_rate": [9e-05],
    "end_epoch": [150],
    "coupling_dim": [512],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

sp_mofop = {
    'n_gpus': [1],
    'method': ['mofop'],
    'max_beta': [1],
    'min_beta': [0],
    "beta_warmup": [50],
    'class_dim': [256],
    "num_mods": [3],
    "num_flows": [5, 10],
    "end_epoch": [1500],
    "coupling_dim": [512],
    "eval_freq": [100],
}

sp_mofop_noflow = {
    'n_gpus': [1],
    'method': ['mofop'],
    'beta': [1.],
    'class_dim': [256],
    "num_mods": [3],
    "num_flows": [0],
    "end_epoch": [500],
    "coupling_dim": [512],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

sp_mogfm = {
    'method': ['mogfm'],
    "initial_learning_rate": [0.0009253348001968961],
    'class_dim': [640],
    "min_beta": [0],
    "max_beta": [1.5142062143401498],
    "beta_warmup": [50],
    "num_gfm_flows": [3],
    "coupling_dim": [32],
    "num_mods": [3],
    "end_epoch": [500],
    'gpu_mem': [30000]
}
