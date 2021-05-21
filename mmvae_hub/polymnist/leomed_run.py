# -*- coding: utf-8 -*-
import os
import time

from sklearn.model_selection import ParameterGrid

search_space_joint_elbo = {
    'n_gpus': [1],
    'method': ['joint_elbo'],
    'gpu_mem': [10000],
    'beta': [2.5],
    "num_mods": [3],
    "end_epoch": [900],
}

search_space_moe = {
    'n_gpus': [1],
    'method': ['moe'],
    'gpu_mem': [10000],
    'beta': [2.5],
    "num_mods": [3],
    "end_epoch": [900],
}

search_space_planar_mixture = {
    'n_gpus': [1],
    'method': ['planar_mixture'],
    'gpu_mem': [10000],
    'beta': [1, 2.5],
    "num_mods": [3],
    "num_flows": [5],
    "end_epoch": [900],
    "weighted_mixture": [True]
}

search_space_planar_pfom = {
    'n_gpus': [1],
    'method': ['pfom'],
    'gpu_mem': [10000],
    'beta': [1, 2.5],
    "num_mods": [3],
    "num_flows": [5],
    "end_epoch": [900],
    "weighted_mixture": [True]
}

for search_space in [search_space_joint_elbo, search_space_moe, search_space_planar_mixture]:

    # for search_space in [search_space_text_enc, search_space_class_dim, search_space_img]:
    for params in ParameterGrid(search_space):

        flags = ''.join(
            f'--{k} {v} ' for k, v in params.items() if
            k not in ['n_gpus', 'gpu_mem', 'factorized_representation', 'use_clf']
        )
        if 'gpu_mem' not in params:
            params['gpu_mem'] = 10000
        command = f'bsub -n 1 -W 8:00 -R "rusage[mem=10000,ngpus_excl_p={params["n_gpus"]}]" -R "select[gpu_mtotal0>={params["gpu_mem"] * params["n_gpus"]}]" python polymnist/main_polymnist.py {flags}'
        # add boolean flags
        if 'factorized_representation' in params and params['factorized_representation']:
            command += ' --factorized_representation'
        if 'use_clf' in params and params['use_clf']:
            command += ' --use_clf'

        print(command)
        os.system(command)
        time.sleep(10)
