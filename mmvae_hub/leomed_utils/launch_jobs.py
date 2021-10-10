import os
import time
from pathlib import Path

import numpy as np
import mmvae_hub


def launch_leomed_jobs(which_dataset: str, params: dict) -> None:
    mmvae_hub_dir = Path(mmvae_hub.__file__).parent
    n_cores = 8

    flags = ''.join(
        f'--{k} {v} ' for k, v in params.items() if
        k not in ['n_gpus', 'gpu_mem', 'factorized_representation', 'use_clf']
    )
    if 'gpu_mem' not in params:
        params['gpu_mem'] = 5000
    if 'n_gpus' not in params:
        params['n_gpus'] = 1

    # 100 epochs take about 5G of space
    scratch_space = int((params['end_epoch'] // 100) * 5) // n_cores or 1

    if which_dataset == 'polymnist':
        python_file = mmvae_hub_dir / 'polymnist/main_polymnist.py'
        mem = 700 * params['num_mods']
        if params['method'] == 'mogfm' or params['method'].startswith('iw'):
            num_hours = int(np.round((params['end_epoch'] * 10) / 60 * 0.5 * params['num_mods'])) or 1
        else:
            # 1 epochs needs approx. 2 minutes
            num_hours = int(np.round((params['end_epoch'] * 2) / 60 * params['num_mods'])) or 1
        # 100 epochs take about 5G of space
        scratch_space = int(np.ceil(((params['end_epoch'] / 100) * 5) / n_cores))

    elif which_dataset == 'mimic':
        python_file = mmvae_hub_dir / 'mimic/main_mimic.py'
        # 1 epochs needs approx. 15 minutes
        num_hours = int(np.round((params['end_epoch'] * 15) / 60)) or 1
        mem = 2500
        # 100 epochs take about 10G of space
        scratch_space = int(np.ceil(((params['end_epoch'] / 100) * 10) / n_cores))

    elif which_dataset == 'celeba':
        python_file = mmvae_hub_dir / 'celeba/main_celeba.py'
        # 1 epochs needs approx. 15 minutes
        num_hours = int(np.round((params['end_epoch'] * 15) / 60)) or 1
        mem = 2500
        # 100 epochs take about 10G of space
        scratch_space = int(np.ceil(((params['end_epoch'] / 100) * 10) / n_cores))

    elif which_dataset == 'mnistsvhntext':
        python_file = mmvae_hub_dir / 'mnistsvhntext/main_svhnmnist.py'

        mem = 2500
        if params['method'] == 'mogfm' or params['method'].startswith('iw'):
            num_hours = int(np.round((params['end_epoch'] * 10) / 60)) or 1
        else:
            # 1 epochs needs approx. 10 minutes
            num_hours = int(np.round((params['end_epoch'] * 10) / 60)) or 1
        # 100 epochs take about 5G of space
        scratch_space = int(np.ceil(((params['end_epoch'] / 100) * 5) / n_cores))

    command = f'bsub -n {n_cores} -W {num_hours}:00 -R "rusage[mem={mem},ngpus_excl_p={params["n_gpus"]},scratch={scratch_space}]" ' \
              f'-R "select[gpu_mtotal0>={params["gpu_mem"] * params["n_gpus"]}]" ' \
              f'python {python_file} {flags}'

    # add boolean flags
    if 'factorized_representation' in params and params['factorized_representation']:
        command += ' --factorized_representation'
    if 'use_clf' in params and params['use_clf']:
        command += ' --use_clf'

    print(command)
    os.system(command)
    time.sleep(1)
