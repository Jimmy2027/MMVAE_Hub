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
    'beta': [1.76],
    'class_dim': [256],
    "num_mods": [3],
    "num_flows": [1],
    "initial_learning_rate": [9e-05],
    "end_epoch": [100],
    "coupling_dim": [512],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

sp_mofop = {
    'n_gpus': [1],
    'method': ['mofop'],
    'beta': [1.],
    'class_dim': [256],
    "num_mods": [3],
    "num_flows": [5],
    "end_epoch": [100],
    "coupling_dim": [512],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}