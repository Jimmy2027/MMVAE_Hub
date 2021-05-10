# -*- coding: utf-8 -*-
import json
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.polymnist.experiment import PolymnistExperiment


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


def show_generated_figs(experiment_dir: Path):
    flags = torch.load(experiment_dir / 'flags.rar')
    if Path(flags.dir_data).name == 'polymnist':
        exp = PolymnistExperiment(flags)

    exp.mm_vae = exp.experiments_database.load_networks_from_db(exp.mm_vae)

    # latest_checkpoint = max(d.name for d in (experiment_dir / 'checkpoints').iterdir() if d.name.startswith('0'))
    #
    # print(f'loading checkpoint from epoch {latest_checkpoint}.')
    #
    # latest_checkpoint_path = experiment_dir / 'checkpoints' / latest_checkpoint
    # exp.mm_vae.load_networks(latest_checkpoint_path)
    plots = generate_plots(exp, epoch=0)

    for p_key, ps in plots.items():
        for name, fig in ps.items():
            plt.figure(figsize=(10, 10))
            plt.imshow(fig)
            plt.title(p_key + '_' + name)
            plt.show()


if __name__ == '__main__':
    # show_generated_figs(Path('/mnt/data/hendrik/mmvae_hub/experiments/polymnist_joint_elbo_2021_05_01_12_20_00_169344'))
    experiment_dir = Path('/mnt/data/hendrik/mmvae_hub/experiments/polymnist_planar_mixture_2021_04_29_23_06_00_937191')
    tensorboard_logs_dir = experiment_dir / 'logs'
    logs_dict = get_logs_dict(tensorboard_logs_dir)
    plot_logs('Generation', logs_dict, together=True)
