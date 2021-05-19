# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt

from mmvae_hub.base.utils.flags_utils import get_experiment
from mmvae_hub.base.utils.plotting import generate_plots


def write_experiment_vis_config(experiment_dir: Path) -> Path:
    """Write a json config that will be read by the experiment_vis jupyter notebook."""
    config = {'experiment_dir': str(experiment_dir)}
    out_path = Path(__file__).parent / f'{experiment_dir.stem}.json'
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    return out_path


def plot_lr_accuracy(logs_dict: dict) -> None:
    lr_accuracy_values = {}
    epochs = [epoch for epoch in logs_dict['epoch_results'] if
              logs_dict['epoch_results'][epoch]['test_results']['lr_eval']]

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        if epoch_values['test_results']['lr_eval']:
            for k, v in epoch_values['test_results']['lr_eval'].items():
                if k not in lr_accuracy_values:
                    lr_accuracy_values[k] = [v['accuracy']]
                else:
                    lr_accuracy_values[k].append(v['accuracy'])

    plt.figure(figsize=(15, 10))
    plt.title('Latent Representation Accuracy')
    for subset, values in lr_accuracy_values.items():
        plt.plot(epochs, values)
    plt.legend([s for s in lr_accuracy_values])
    plt.show()


def plot_likelihoods(logs_dict: dict) -> None:
    lhoods = {}
    epochs = [epoch for epoch in logs_dict['epoch_results'] if
              logs_dict['epoch_results'][epoch]['test_results']['lhoods']]

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        if epoch_values['test_results']['lhoods']:
            for subset, v_subset in epoch_values['test_results']['lhoods'].items():
                if subset not in lhoods:
                    lhoods[subset] = {}
                for mod, result in v_subset.items():
                    if mod in lhoods[subset]:
                        lhoods[subset][mod].append(result)
                    else:
                        lhoods[subset][mod] = [result]

    for subset, v in lhoods.items():
        plt.figure(figsize=(10, 5))
        plt.title(f'Likelihoods for subset {subset}.')
        for mod, values in v.items():
            plt.plot(epochs, values)
        plt.legend([s for s in v])
        plt.show()


def plot_basic_batch_logs(phase: str, logs_dict: dict):
    """
    phase: either train or test
    """
    results_dict = {'total_loss': [], 'klds': {}, 'log_probs': {}, 'joint_divergence': []}

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        v = epoch_values[f'{phase}_results']
        results_dict['total_loss'].append(v['total_loss'])
        results_dict['joint_divergence'].append(v['joint_divergence'])

        for log_k in ['klds', 'log_probs']:
            for s_key, s_value in v[log_k].items():
                if s_key not in results_dict[log_k]:
                    results_dict[log_k][s_key] = [s_value]
                else:
                    results_dict[log_k][s_key].append(s_value)

    for k in ['total_loss', 'joint_divergence']:
        plt.figure(figsize=(10, 5))
        plt.title(k)
        plt.plot(results_dict[k])
        plt.legend(k)
        plt.show()

    for k in ['klds', 'log_probs']:
        plt.figure(figsize=(10, 5))
        plt.title(k)
        for subset, values in results_dict[k].items():
            plt.plot(values)
        plt.legend([s for s in results_dict[k]])
        plt.show()


def show_latents(logs_dict: dict) -> None:
    enc_mods_mus = {}
    enc_mods_logvars = {}

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        if epoch_values['test_results']['latents']:
            for mod_str, v in epoch_values['test_results']['latents'].items():
                if mod_str not in enc_mods_mus:
                    enc_mods_mus[mod_str] = [v['latents_class']['mu']]
                    enc_mods_logvars[mod_str] = [v['latents_class']['logvar']]
                else:
                    enc_mods_mus[mod_str].append(v['latents_class']['mu'])
                    enc_mods_logvars[mod_str].append(v['latents_class']['logvar'])

    for k, d in {'mus': enc_mods_mus, 'logvars': enc_mods_logvars}.items():
        plt.figure(figsize=(10, 5))
        plt.title(f'Latent {k}.')
        for mod_str, values in d.items():
            plt.plot(values)
        plt.legend([s for s in d])
        plt.show()


def plot_coherence_accuracy(logs_dict: dict) -> None:
    gen_eval_logs = {}
    epochs = [epoch for epoch in logs_dict['epoch_results'] if
              logs_dict['epoch_results'][epoch]['test_results']['gen_eval']]

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        if epoch_values['test_results']['gen_eval']:
            for k, v in epoch_values['test_results']['gen_eval'].items():
                k = k.removeprefix('digit_')
                num_input_mods = len(k.split('__')[0].split('_'))
                if num_input_mods not in gen_eval_logs:
                    gen_eval_logs[num_input_mods] = {k: [v]}
                elif k not in gen_eval_logs[num_input_mods]:
                    gen_eval_logs[num_input_mods][k] = [v]
                else:
                    gen_eval_logs[num_input_mods][k].append(v)

    for num_input_mods, v in gen_eval_logs.items():
        plt.figure(figsize=(10, 5))
        plt.title(f'Gen eval Accuracy with {num_input_mods} input modalities.')
        for subset, values in v.items():
            plt.plot(epochs, values)
        plt.legend([s for s in v])
        plt.show()


def show_generated_figs(experiment_dir: Path = None, flags=None):
    if not flags:
        flags = torch.load(experiment_dir / 'flags.rar')
    if not hasattr(flags, 'weighted_mixture'):
        flags.weighted_mixture = False
    exp = get_experiment(flags)

    if experiment_dir and (experiment_dir / 'checkpoints').exists():
        latest_checkpoint = max(int(d.name) for d in (experiment_dir / 'checkpoints').iterdir() if d.name.isdigit())

        print(f'loading checkpoint from epoch {latest_checkpoint}.')

        latest_checkpoint_path = experiment_dir / 'checkpoints' / str(latest_checkpoint).zfill(4)
        exp.mm_vae.load_networks(latest_checkpoint_path)
    else:
        # load networks from database
        exp.mm_vae = exp.experiments_database.load_networks_from_db(exp.mm_vae)

    plots = generate_plots(exp, epoch=0)

    for p_key, ps in plots.items():
        for name, fig in ps.items():
            plt.figure(figsize=(10, 10))
            plt.imshow(fig)
            plt.title(p_key + '_' + name)
            plt.show()


def make_experiments_dataframe(experiments):
    df = pd.DataFrame()
    for exp in experiments.find({}):
        if exp['epoch_results'] is not None and exp['epoch_results']:
            last_epoch = str(max(int(epoch) for epoch in exp['epoch_results']))
            last_epoch_results = exp['epoch_results'][last_epoch]['test_results']

            if exp['flags']['method'] == 'planar_mixture':
                exp['flags']['method'] = f"{exp['flags']['method']}_{exp['flags']['num_flows']}"

            if last_epoch_results['lr_eval'] and last_epoch_results['gen_eval']:

                results_dict = {**exp['flags'], 'end_epoch': last_epoch, '_id': exp['_id']}

                scores = []
                scores_lr = []
                # get lr_eval results
                for key, val in last_epoch_results['lr_eval'].items():
                    results_dict[f'lr_eval_{key}'] = val['accuracy']
                    scores.append(val['accuracy'])
                    scores_lr.append(val['accuracy'])

                scores_gen = []
                # get gen_eval results
                for key, val in last_epoch_results['gen_eval'].items():
                    key = key.replace('digit_', '')
                    results_dict[f'gen_eval_{key}'] = val
                    scores.append(val)
                    scores_gen.append(val)

                scores_prd = []

                if 'prd_scores' in last_epoch_results and last_epoch_results['prd_scores']:
                    for key, val in last_epoch_results['prd_scores'].items():
                        results_dict[f'prd_score_{key}'] = val
                        scores_prd.append(val)

                results_dict['score'] = sum(scores)
                results_dict['score_lr'] = sum(scores_lr)
                results_dict['score_gen'] = sum(scores_gen)
                results_dict['score_prd'] = sum(scores_prd)
                df = df.append(results_dict, ignore_index=True)
    return df


if __name__ == '__main__':
    # show_generated_figs(Path('/mnt/data/hendrik/mmvae_hub/experiments/polymnist_joint_elbo_2021_05_01_12_20_00_169344'))
    experiment_dir = Path('/mnt/data/hendrik/mmvae_hub/experiments/polymnist_planar_mixture_2021_04_29_23_06_00_937191')
    tensorboard_logs_dir = experiment_dir / 'logs'
    logs_dict = get_logs_dict(tensorboard_logs_dir)
    plot_logs('Generation', logs_dict, together=True)
