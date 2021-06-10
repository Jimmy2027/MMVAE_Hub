# -*- coding: utf-8 -*-
import os
import time

from sklearn.model_selection import ParameterGrid

search_spaces_poe = {
    'method': ['poe'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

search_spaces_moe = {
    'method': ['moe'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}

search_spaces_je = {
    'method': ['joint_elbo'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}
search_spaces_pm = {
    'method': ['planar_mixture'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}
search_spaces_pfom = {
    'method': ['pfom'],
    'class_dim': [256],
    "beta": [1],
    "num_flows": [5],
    "num_mods": [3],
    "end_epoch": [100],
    "weighted_mixture": [False],
    "amortized_flow": [False]
}


# for search_space in [search_space_joint_elbo, search_space_moe, search_space_planar_mixture]:
for search_space in [search_spaces_poe, search_spaces_moe, search_spaces_je, search_spaces_pm, search_spaces_pfom]:

    for params in ParameterGrid(search_space):

        flags = ''.join(
            f'--{k} {v} ' for k, v in params.items() if
            k not in ['n_gpus', 'gpu_mem', 'factorized_representation', 'use_clf']
        )
        if 'gpu_mem' not in params:
            params['gpu_mem'] = 5000
        if 'n_gpus' not in params:
            params['n_gpus'] = 1

        # 100 epochs need a bit less than 2 hours
        num_hours = int((params['end_epoch'] // 100) * 2) or 1

        # 100 epochs take about 5G of space
        scratch_space = int((params['end_epoch'] // 100) * 5) // 8

        if 'optuna' in params and params['optuna']:
            python_file = 'hyperopt/HyperoptTrainer.py'
        else:
            python_file = 'polymnist/main_polymnist.py'

        command = f'bsub -n 8 -W {num_hours}:00 -R "rusage[mem=1000,ngpus_excl_p={params["n_gpus"]},scratch={scratch_space}]" ' \
                  f'-R "select[gpu_mtotal0>={params["gpu_mem"] * params["n_gpus"]}]" ' \
                  f'python {python_file} {flags}'

        # add boolean flags
        if 'factorized_representation' in params and params['factorized_representation']:
            command += ' --factorized_representation'
        if 'use_clf' in params and params['use_clf']:
            command += ' --use_clf'

        print(command)
        os.system(command)
        time.sleep(10)
