# -*- coding: utf-8 -*-
import glob
import json
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from tensorflow.compat.v1.train import summary_iterator

from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.polymnist.experiment import PolymnistExperiment


def write_experiment_vis_config(experiment_dir: Path) -> Path:
    """Write a json config that will be read by the experiment_vis jupyter notebook."""
    config = {'experiment_dir': str(experiment_dir)}
    out_path = Path(__file__).parent / f'{experiment_dir.stem}.json'
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    return out_path


def plot_lr_accuracy(tensorboard_logs_dir: Path) -> None:
    lr_logs = {}
    for lr_logdir in glob.glob(f'{tensorboard_logs_dir}/Latent Representation*'):
        dir_name = Path(lr_logdir).stem
        if dir_name.endswith('accuracy'):
            # get subset from dir_name
            subset = dir_name[len('Latent Representation_'):][:-len('_accuracy')]
            lr_logs[subset] = []
            for e in summary_iterator(glob.glob(f'{lr_logdir}/events.out.tfevents*')[0]):
                for v in e.summary.value:
                    lr_logs[subset].append(v.simple_value)

    plt.figure()
    plt.title('Latent Representation Accuracy')
    for subset, values in lr_logs.items():
        plt.plot(values)
    plt.legend([s for s in lr_logs])
    plt.show()


def plot_gen_results(tensorboard_logs_dir: Path) -> None:
    lr_logs = {}
    for lr_logdir in glob.glob(f'{tensorboard_logs_dir}/Generation*'):
        for e in summary_iterator(glob.glob(f'{lr_logdir}/events.out.tfevents*')[0]):
            if e.summary.value:
                tag = e.summary.value[0].tag
                if tag not in lr_logs:
                    lr_logs[tag] = []
                for v in e.summary.value:
                    lr_logs[tag].append(v.simple_value)

    plt.figure()
    plt.title('Gen Eval')
    for subset, values in lr_logs.items():
        plt.plot(values)
    plt.legend([s.split('/')[-1] for s in lr_logs])
    plt.show()


def get_logs_dict(tensorboard_logs_dir: Path):
    logs_dict = {}
    for tf_logs in glob.glob(f'{tensorboard_logs_dir}/*/*'):
        for e in summary_iterator(tf_logs):
            for v in e.summary.value:
                tag = v.tag
                if tag not in logs_dict:
                    logs_dict[tag] = []
                logs_dict[tag].append(v.simple_value)

    return logs_dict


def plot_logs(category: str, logs_dict: dict, together: bool = False):
    """
    together: bool indicating if plots should be plotted individually or all in one plot.
    """
    cath_logs = {k.split('/')[-1]: v for k, v in logs_dict.items() if k.startswith(category)}
    if together:
        for k in logs_dict:
            # fixme
            break
        title = k.split('/')[:-1]
        plt.figure()
        plt.title(title)
        for k, v in cath_logs.items():
            plt.plot(v)
        plt.legend([k for k in cath_logs])
        plt.show()
    else:
        for k, v in cath_logs.items():
            plt.plot(v)
            plt.title(k)
            plt.show()


def show_generated_figs(experiment_dir: Path):
    flags = torch.load(experiment_dir / 'flags.rar')
    if Path(flags.dir_data).name == 'polymnist':
        exp = PolymnistExperiment(flags)
    latest_checkpoint = max(d.name for d in (experiment_dir / 'checkpoints').iterdir() if d.name.startswith('0'))

    print(f'loading checkpoint from epoch {latest_checkpoint}.')

    latest_checkpoint_path = experiment_dir / 'checkpoints' / latest_checkpoint / 'mm_vae'
    exp.mm_vae.load_networks(latest_checkpoint_path)
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
