"""Base search spaces to compare between versions."""

base_params = {
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

base_search_spaces = [{'method': [method], **base_params} for method in
                      ['poe', 'moe', 'joint_elbo', 'planar_mixture', 'pfom']]

search_space_poe = {
    'method': ['poe'],
    **base_params
}

search_space_moe = {
    'method': ['moe'],
    **base_params
}

search_space_je = {
    'method': ['joint_elbo'],
    "num_mods": [3],
    "end_epoch": [2000],
    'initial_learning_rate': [0.0009439],
    'class_dim': [128],
    'beta': [0.34]
}
search_space_pm = {
    'method': ['planar_mixture'],
    **base_params
}
search_space_pfom = {
    'method': ['pfom'],
    **base_params
}

search_space_gfm = {
    'method': ['gfmop'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [1],
    "num_mods": [3],
    "end_epoch": [2000],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

search_space_mofop = {
    'method': ['mofop'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [1],
    "num_mods": [3],
    "end_epoch": [2000],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

search_space_gfmop = {
    'method': ['gfmop'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [1, 5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}
