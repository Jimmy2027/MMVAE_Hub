# -*- coding: utf-8 -*-
import glob
import json
from pathlib import Path

from matplotlib import pyplot as plt
from tensorflow.compat.v1.train import summary_iterator


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


def plot_logs(cathegory: str, logs_dict: dict, together: bool = False):
    cath_logs = {k.split('/')[-1]: v for k, v in logs_dict.items() if k.startswith(cathegory)}
    if together:
        for k in logs_dict:
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
