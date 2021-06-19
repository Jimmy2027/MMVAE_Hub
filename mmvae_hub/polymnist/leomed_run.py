# -*- coding: utf-8 -*-
import os
import time

from sklearn.model_selection import ParameterGrid

from mmvae_hub.hyperopt.search_spaces.base_search_spaces import base_search_spaces

for search_space in [base_search_spaces]:

    for params in ParameterGrid(search_space):

        n_cores = 8

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
        scratch_space = int((params['end_epoch'] // 100) * 5) // n_cores or 1

        if 'optuna' in params and params['optuna']:
            python_file = 'hyperopt/HyperoptTrainer.py'
        else:
            python_file = 'polymnist/main_polymnist.py'

        command = f'bsub -n {n_cores} -W {num_hours}:00 -R "rusage[mem=1500,ngpus_excl_p={params["n_gpus"]},scratch={scratch_space}]" ' \
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
