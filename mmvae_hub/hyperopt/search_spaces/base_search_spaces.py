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
    **base_params
}
search_space_pm = {
    'method': ['planar_mixture'],
    **base_params
}
search_space_pfom = {
    'method': ['pfom'],
    **base_params
}

base_search_spaces = [search_space_pfom, search_space_pm, search_space_je, search_space_poe, search_space_moe]
