import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from mmvae_hub import log


def create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name)


def get_experiment_uid(exp_str_prefix: str, method: str) -> str:
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f")
    return exp_str_prefix + '_' + method + '_' + dateStr


def create_dir_structure_testing(exp):
    flags = exp.flags
    for k, label_str in enumerate(exp.labels):
        dir_gen_eval_label = os.path.join(flags.dir_gen_eval, label_str)
        create_dir(dir_gen_eval_label)
        dir_inference_label = os.path.join(flags.dir_inference, label_str)
        create_dir(dir_inference_label)


def create_dir_structure(flags: argparse.ArgumentParser(), train: bool = True) -> argparse.ArgumentParser:
    """
    Create following directories under flags.dir_experiment:
    experiment_uid, checkpoints, logs, generation_evaluation, inference, fid_eval, plots.

    Under flags.dir_experiment/plots, create:
    cond_gen, random_samples, swapping
    """
    experiment_uid = flags.experiment_uid
    if train:
        flags.dir_experiment_run = flags.dir_experiment / experiment_uid
        flags.str_experiment = experiment_uid
    else:
        flags.dir_experiment_run = (flags.dir_experiment).expanduser()

    log.info(f'dir_experiment_run: {flags.dir_experiment_run}')
    if train:
        create_dir(flags.dir_experiment_run)

    flags.dir_checkpoints = Path(flags.dir_experiment_run) / 'checkpoints'
    if train:
        create_dir(os.path.expanduser(flags.dir_checkpoints))

    flags.dir_logs = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'logs')
    if train:
        create_dir(flags.dir_logs)

    flags.dir_gen_eval = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'generation_evaluation')
    if train:
        create_dir(flags.dir_gen_eval)

    flags.dir_inference = os.path.join(os.path.expanduser(flags.dir_experiment_run), 'inference')
    if train:
        create_dir(flags.dir_inference)

    if not train:
        flags.dir_fid = os.path.join(flags.dir_experiment_run, 'fid_eval')
        if not os.path.exists(flags.dir_fid):
            os.makedirs(flags.dir_fid)
    flags.dir_gen_eval_fid = os.path.join(flags.dir_fid, experiment_uid)
    log.info(f'dir_gen_eval_fid: {flags.dir_gen_eval_fid}')
    create_dir(flags.dir_gen_eval_fid)

    flags.dir_plots = os.path.join(flags.dir_experiment_run, 'plots')
    if train:
        create_dir(flags.dir_plots)
    flags.dir_swapping = os.path.join(flags.dir_plots, 'swapping')
    if train:
        create_dir(flags.dir_swapping)

    flags.dir_random_samples = os.path.join(flags.dir_plots, 'random_samples')
    if train:
        create_dir(flags.dir_random_samples)

    flags.dir_cond_gen = os.path.join(flags.dir_plots, 'cond_gen')
    if train:
        create_dir(flags.dir_cond_gen)

    return flags


def expand_paths(flags: argparse.ArgumentParser()) -> argparse.ArgumentParser():
    flags.dir_data = os.path.expanduser(flags.dir_data)
    flags.dir_experiment = os.path.expanduser(flags.dir_experiment)
    flags.inception_state_dict = os.path.expanduser(flags.inception_state_dict)
    flags.dir_fid = os.path.expanduser(flags.dir_fid)
    flags.dir_clf = os.path.expanduser(flags.dir_clf)
    return flags
