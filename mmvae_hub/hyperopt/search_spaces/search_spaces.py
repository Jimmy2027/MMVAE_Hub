search_space_joint_elbo = {
    'n_gpus': [1],
    'method': ['joint_elbo'],
    'beta': [2.5],
    "num_mods": [3],
    "end_epoch": [900],
}

search_space_moe = {
    'n_gpus': [1],
    'method': ['moe'],
    'beta': [2.5],
    "num_mods": [3],
    "end_epoch": [900],
}

search_space_planar_mixture = {
    'n_gpus': [1],
    'method': ['planar_mixture'],
    'beta': [1],
    'class_dim': [256],
    "num_mods": [3],
    "num_flows": [5],
    "end_epoch": [2000],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

sp_pfom = {
    'n_gpus': [1],
    'method': ['pfom'],
    'beta': [1],
    'class_dim': [256],
    "num_mods": [3],
    "num_flows": [5],
    "initial_learning_rate": [0.005],
    "end_epoch": [2000],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

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
