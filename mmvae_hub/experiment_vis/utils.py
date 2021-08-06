# -*- coding: utf-8 -*-
import functools
import json
import operator
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import nbformat
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from matplotlib import pyplot as plt
from mmvae_hub import log
from mmvae_hub.mimic.experiment import MimicExperiment
from mmvae_hub.modalities import BaseModality
from mmvae_hub.networks.BaseMMVae import BaseMMVAE
from mmvae_hub.networks.FlowVaes import FlowVAE, FoMoP
from mmvae_hub.polymnist.experiment import PolymnistExperiment
from mmvae_hub.utils.Dataclasses import ReparamLatent
from mmvae_hub.utils.MongoDB import MongoDatabase
from mmvae_hub.utils.plotting.plotting import generate_plots
from mmvae_hub.utils.setup.flags_utils import BaseFlagsSetup, get_config_path
from mmvae_hub.utils.utils import dict2json
from nbconvert import HTMLExporter, PDFExporter
from nbconvert.preprocessors import ExecutePreprocessor
from pandas import DataFrame

FLOW_METHODS = ['planar_mixture', 'pfom', 'pope', 'fomfop', 'fomop', 'mofop', 'planar_vae', 'mofogfm']


def run_notebook_convert(dir_experiment_run: Path = None) -> Path:
    """
    Run and convert the notebook to html and pdf.
    """
    # Copy the experiment_vis jupyter notebook to the experiment dir
    notebook_path = Path(__file__).parent.parent / 'experiment_vis/experiment_vis.ipynb'
    dest_notebook_path = dir_experiment_run / 'experiment_vis.ipynb'

    # copy notebook to experiment run
    shutil.copyfile(notebook_path, dest_notebook_path)

    log.info(f'Executing experiment vis notebook {dest_notebook_path}.')
    with open(dest_notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': str(dest_notebook_path.parent)}})

    nbconvert_path = dest_notebook_path.with_suffix('.nbconvert.ipynb')

    with open(nbconvert_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    log.info('Converting notebook to html.')
    html_path = nbconvert_path.with_suffix('.html')
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'
    (body, resources) = html_exporter.from_notebook_node(nb)
    with open(html_path, 'w') as f:
        f.write(body)

    log.info('Converting notebook to pdf.')
    pdf_path = nbconvert_path.with_suffix('.pdf')
    pdf_exporter = PDFExporter()
    pdf_exporter.template_name = 'classic'
    (body, resources) = pdf_exporter.from_notebook_node(nb)
    pdf_path.write_bytes(body)

    return pdf_path


def upload_notebook_to_db(experiment_uid: str) -> None:
    """
    Run the experiment vis notebook and upload it with ppb to db.
    """
    import ppb

    with tempfile.TemporaryDirectory() as tmpdirname:
        dir_experiment_run = Path(tmpdirname) / experiment_uid
        dir_experiment_run.mkdir()

        db = MongoDatabase(training=False, _id=experiment_uid)
        dict2json(dir_experiment_run / 'flags.json', db.get_experiment_dict()['flags'])

        pdf_path = run_notebook_convert(dir_experiment_run=dir_experiment_run)

        expvis_url = ppb.upload(pdf_path, plain=True)
        log.info(f'Experiment_vis was uploaded to {expvis_url}')
        db.insert_dict({'expvis_url': expvis_url})


def write_experiment_vis_config(experiment_dir: Path) -> Path:
    """Write a json config that will be read by the experiment_vis jupyter notebook."""
    config = {'experiment_dir': str(experiment_dir)}
    out_path = Path(__file__).parent / f'{experiment_dir.stem}.json'
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    return out_path


def get_epochs(logs_dict: dict, metric: str):
    """Get all the epochs the metric was evaluated."""
    return sorted([int(epoch) for epoch in logs_dict['epoch_results'] if
                   logs_dict['epoch_results'][epoch]['test_results'] and
                   logs_dict['epoch_results'][epoch]['test_results'][metric] is not None])


def plot_lr_accuracy(logs_dict: dict) -> None:
    """
    Plot the latent representation evaluation accuracy.
    For method planar mixture, show both evaluations, before and after applying flows.
    """
    method = logs_dict['flags']['method']
    lr_accuracy_values = {}

    # for backwards compatibility:
    if (
            logs_dict['epoch_results']['0']['test_results'] is not None
            and 'lr_eval_q0' not in logs_dict['epoch_results']['0']['test_results']
    ):
        for epoch in logs_dict['epoch_results']:
            if logs_dict['epoch_results'][epoch]['test_results'] is None:
                logs_dict['epoch_results'][epoch]['test_results'] = {'lr_eval': None}
            test_results = logs_dict['epoch_results'][epoch]['test_results']
            if test_results['lr_eval'] is not None:
                if method in ['planar_mixture', 'pfom']:
                    test_results['lr_eval_zk'] = test_results['lr_eval']
                    test_results['lr_eval_q0'] = {k: {'accuracy': None} for k in test_results['lr_eval']}
                else:
                    test_results['lr_eval_q0'] = test_results['lr_eval']
                logs_dict['epoch_results'][epoch]['test_results'] = test_results
            else:
                logs_dict['epoch_results'][epoch]['test_results']['lr_eval_q0'] = None
                logs_dict['epoch_results'][epoch]['test_results']['lr_eval_zk'] = None

    epochs = get_epochs(logs_dict, metric='lr_eval_q0' if method != 'planar_mixture' else 'lr_eval_zk')

    if method in ['pfom', 'mofop']:
        lr_keys = ['q0', 'zk']
    elif method == 'planar_mixture':
        lr_keys = ['zk']
    else:
        lr_keys = ['q0']

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        for lr_key in lr_keys:
            if lr_key not in lr_accuracy_values:
                lr_accuracy_values[lr_key] = {}
            if epoch_values['test_results'] and epoch_values['test_results'][f'lr_eval_{lr_key}']:
                for k, v in epoch_values['test_results'][f'lr_eval_{lr_key}'].items():
                    # insert accuracy value to dict
                    if k not in lr_accuracy_values[lr_key]:
                        lr_accuracy_values[lr_key][k] = [v['accuracy']]
                    else:
                        lr_accuracy_values[lr_key][k].append(v['accuracy'])

    if method in ['pfom', 'mofop']:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Latent Representation Accuracy')
        for ax_idx, (lr_key, lr_values) in enumerate(lr_accuracy_values.items()):
            for subset, values in lr_values.items():
                axs[ax_idx].plot(epochs, values)
            axs[ax_idx].legend([s for s in lr_values])
            axs[ax_idx].set_title(lr_key)
    else:
        lr_key = lr_keys[0]
        plt.figure(figsize=(15, 10))
        plt.title('Latent Representation Accuracy')
        for subset, values in lr_accuracy_values[lr_key].items():
            plt.plot(epochs, values)
        plt.legend([s for s in lr_accuracy_values[lr_key]])
    plt.show()


def plot_likelihoods(logs_dict: dict) -> None:
    lhoods = {}
    epochs = get_epochs(logs_dict, 'lhoods')

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        if epoch_values['test_results'] and epoch_values['test_results']['lhoods']:
            for subset, v_subset in epoch_values['test_results']['lhoods'].items():
                if subset not in lhoods:
                    lhoods[subset] = {}
                for mod, result in v_subset.items():
                    if mod in lhoods[subset]:
                        lhoods[subset][mod].append(result)
                    else:
                        lhoods[subset][mod] = [result]

    for subset, v in lhoods.items():
        plt.figure(figsize=(15, 5))
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
        if v:
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
        if epoch_values['test_results'] and epoch_values['test_results']['latents']:
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
    epochs = get_epochs(logs_dict, 'gen_eval')

    for epoch, epoch_values in logs_dict['epoch_results'].items():
        if epoch_values['test_results'] and epoch_values['test_results']['gen_eval']:
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


def plot_prd_scores(df):
    prd_scores = {}

    for col in df.columns:
        if col.startswith('prd'):
            if 'random' in col:
                key = 'random'
            else:
                num_in_mods = len(col.replace('prd_score_', '').split('_')) - 1
                key = f'{num_in_mods}_in_mods'
            if key not in prd_scores:
                prd_scores[key] = [col]
            else:
                prd_scores[key].append(col)

    for k, prd_s in prd_scores.items():
        df[prd_s].plot.bar(title=k, figsize=(10, 5))
        plt.show()


def plot_betas(logs_dict: dict) -> None:
    betas = []
    epochs = [int(k) for k in logs_dict['epoch_results']]

    if 'min_beta' not in logs_dict['flags']:
        betas = [logs_dict['flags']['beta'] for _ in epochs]
    else:
        for epoch, epoch_values in logs_dict['epoch_results'].items():
            if 'beta' in epoch_values:
                betas.append(epoch_values['beta'])

    x_ticks = epochs[0::int(len(epochs) / 15)]
    x_ticks.append(epochs[-1])
    plt.plot(epochs, betas)
    plt.xticks(x_ticks)
    plt.legend(['beta'])
    plt.show()


def save_cond_gen(_id: str, save_path: Path):
    """
    Save examples of the cond gen to save_path.
    """
    exp = load_experiment(_id=_id)
    exp.set_eval_mode()
    model = exp.mm_vae
    subsets = exp.subsets
    mods: Mapping[str, BaseModality] = exp.modalities

    test_samples = exp.test_samples

    # get style from random sampling
    random_styles = model.get_random_styles(1)

    input_samples = {}

    # save input images
    for _, mod in mods.items():
        input_sample = mod.plot_data(test_samples[0][mod.name])
        input_samples[mod.name] = input_sample
        plt.imshow(input_sample.cpu().moveaxis(0, -1))
        plt.axis('off')
        savefig_path = save_path / 'input_samples' / f'{mod.name}.png'
        savefig_path.parent.mkdir(exist_ok=True, parents=True)
        plt.title(f'{mod.name}', fontsize=30)
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)
        # plt.show()

    for s_key in subsets:
        subset = subsets[s_key]
        i_batch = {
            mod.name: input_samples[mod.name].unsqueeze(0)
            for mod in subset
        }
        # infer latents from batch
        enc_mods, joint_latent = model.inference(i_batch)

        # c_rep is embedding of subset
        c_rep = joint_latent.get_subset_embedding(s_key)

        style = {}
        for m_key_out in mods:
            mod_out = mods[m_key_out]
            if exp.flags.factorized_representation:
                style[mod_out.name] = random_styles[mod_out.name][i].unsqueeze(0)
            else:
                style[mod_out.name] = None
        cond_mod_in = ReparamLatent(content=c_rep, style=style)
        cond_gen_samples = model.generate_from_latents(cond_mod_in)

        for out_key, out_plot in cond_gen_samples.items():
            plt.imshow(out_plot.squeeze().detach().cpu().moveaxis(0, -1))
            plt.axis('off')
            savefig_path = save_path / f'{s_key}' / f'{out_key}.png'
            savefig_path.parent.mkdir(exist_ok=True, parents=True)
            plt.title(f'{s_key} -> {out_key}', fontsize=30)
            plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)
            # plt.show()


def load_experiment(experiment_dir: Path = None, flags=None, _id: str = None):
    """Load experiment with flags and trained model from old experiment."""
    if not flags:
        dataset = _id.split('_')[0].lower()
        flags_setup = BaseFlagsSetup(get_config_path(dataset=dataset))
        flags_path = Path('flags.rar')
        if flags_path.exists():
            flags = flags_setup.load_old_flags(flags_path, add_args={'save_figure': False})
        else:
            flags = flags_setup.load_old_flags(_id=_id, add_args={'save_figure': False})
    exp = get_experiment(flags)

    if experiment_dir and (experiment_dir / 'checkpoints').exists():
        latest_checkpoint = max(int(d.name) for d in (experiment_dir / 'checkpoints').iterdir() if d.name.isdigit())

        print(f'loading checkpoint from epoch {latest_checkpoint}.')

        latest_checkpoint_path = experiment_dir / 'checkpoints' / str(latest_checkpoint).zfill(4)
        exp.mm_vae.load_networks(latest_checkpoint_path)
    else:
        # load networks from database
        exp.mm_vae = exp.experiments_database.load_networks_from_db(exp.mm_vae)

    return exp


def show_generated_figs(experiment_dir: Path = None, flags=None, _id: str = None, return_plots: bool = False,
                        nbr_samples_x: int = 10, nbr_samples_y: int = 10):
    exp = load_experiment(experiment_dir, flags, _id)

    exp.set_eval_mode()
    plots = generate_plots(exp, epoch=0, nbr_samples_x=nbr_samples_x, nbr_samples_y=nbr_samples_y)

    if return_plots:
        return plots

    for p_key, ps in plots.items():
        for name, fig in ps.items():
            plt.figure(figsize=(10, 10))
            plt.imshow(fig)
            title = p_key + '_' + name
            plt.title(title)
            plt.show()


def bw_compat_epoch_results(epoch_results: dict, method: str, flags: dict):
    """
    Adapt epoch results for backwards compatibility.
    """
    if 'max_beta' not in flags or flags['max_beta'] is None:
        flags['max_beta'] = flags['min_beta'] = flags['beta']
        flags['beta_warmup'] = 0

        # some experiments have 'lr_eval_qk' instead of 'lr_eval_zk'
    if 'lr_eval_qk' in epoch_results:
        epoch_results['lr_eval_zk'] = epoch_results['lr_eval_qk']

    if 'lr_eval_zk' not in epoch_results:
        epoch_results['lr_eval_zk'] = None

    if 'lr_eval_q0' not in epoch_results:
        if method in {'planar_mixture', 'pfom'}:
            epoch_results['lr_eval_zk'] = epoch_results['lr_eval']
            epoch_results['lr_eval_q0'] = {k: {'accuracy': None} for k in epoch_results['lr_eval']}
        else:
            epoch_results['lr_eval_q0'] = epoch_results['lr_eval']

    return epoch_results, flags


def make_experiments_dataframe(experiments):
    """
    Create a dataframe with all experiment results.
    A score is created by summing up all metrics from the lr_eval, the gen_val and the prd_eval. The score is
    normalized by the number of modalities.
    """
    df = pd.DataFrame()
    for exp in experiments.find({}):
        if exp['epoch_results'] is not None and exp['epoch_results']:
            method = exp['flags']['method']

            max_epoch = max(int(epoch) for epoch in exp['epoch_results'])

            # get the last epoch where evaluation was run.
            if (max_epoch - max_epoch % int(exp['flags']['eval_freq']) > 1
                    and exp['epoch_results'][str(max_epoch)]['test_results']['gen_eval'] is None):
                last_epoch = str(max_epoch - max_epoch % int(exp['flags']['eval_freq']) - 1)
            else:
                last_epoch = str(max_epoch)
            last_epoch_results = exp['epoch_results'][last_epoch]['test_results']

            if method not in ['joint_elbo', 'poe', 'moe']:
                exp['flags']['method'] = f"{exp['flags']['method']}_{exp['flags']['num_flows']}"

            # for backwards compatibility:
            last_epoch_results, flags = bw_compat_epoch_results(last_epoch_results, method, exp['flags'])

            # if lr_eval and gen_eval, add results to df.
            if (last_epoch_results['lr_eval_q0'] or last_epoch_results['lr_eval_zk']) \
                    and last_epoch_results['gen_eval']:
                results_dict = {**flags, 'end_epoch': last_epoch, '_id': exp['_id']}
                # try:
                if 'expvis_url' in exp:
                    results_dict['expvis_url'] = exp['expvis_url']

                if method == 'fomop':
                    score_lr, score_lr_q0, score_lr_zk = FoMoP.calculate_lr_eval_scores(last_epoch_results)
                elif method in FLOW_METHODS:
                    score_lr, score_lr_q0, score_lr_zk = FlowVAE.calculate_lr_eval_scores(last_epoch_results)
                else:
                    score_lr, score_lr_q0, score_lr_zk = BaseMMVAE.calculate_lr_eval_scores(last_epoch_results)

                scores_gen = []
                # get gen_eval results
                for key, val in last_epoch_results['gen_eval'].items():
                    key = key.replace('digit_', '')
                    results_dict[f'gen_eval_{key}'] = val
                    scores_gen.append(val)

                scores_prd = []
                # get prd scores
                if 'prd_scores' in last_epoch_results and last_epoch_results['prd_scores']:
                    for key, val in last_epoch_results['prd_scores'].items():
                        results_dict[f'prd_score_{key}'] = val
                        scores_prd.append(val)

                # results_dict['score'] = np.mean(scores)

                results_dict['score_lr_q0'] = score_lr_q0
                results_dict['score_lr_zk'] = score_lr_zk
                results_dict['score_lr'] = results_dict['score_lr_zk'] if method in ['planar_mixture', 'pfom'] \
                    else results_dict['score_lr_q0']
                results_dict['score_gen'] = np.mean(scores_gen)
                results_dict['score_prd'] = np.mean(scores_prd)

                results_dict['score'] = (score_lr / 0.75) + (results_dict['score_gen'] / 0.55) + (
                        results_dict['score_prd'] / 0.1)

                df = df.append(results_dict, ignore_index=True)

                # except Exception as e:
                #     print(e)

        else:
            print(f'skipping experiment {exp["_id"]}')
    return df


def get_experiment(flags):
    """
    Get experiments class from dir_data flag.
    """
    if Path(flags.dir_data).name in ['PolyMNIST', 'polymnist']:
        return PolymnistExperiment(flags)
    elif Path(flags.dir_data).name.startswith('mimic') or Path(flags.dir_data).name == 'scratch' or Path(
            flags.dir_data).name == 'leomed_klugh':
        return MimicExperiment(flags)
    elif flags.dataset == 'toy':
        return PolymnistExperiment(flags)
    else:
        raise RuntimeError(f'No experiment for {Path(flags.dir_data).name} implemented.')


def display_df(df: DataFrame) -> None:
    display(HTML(df.to_html()))


def compare_methods(df, methods: list, df_selectors: dict):
    methods_selection = functools.reduce(operator.or_, (df['method'].str.startswith(method) for method in methods))
    df_selection = functools.reduce(operator.and_, (df[k] == v for k, v in df_selectors.items()))
    display_df(df.loc[(methods_selection) & (df_selection)])


def display_base_params(df, methods: list, show_cols: list, num_flows: int = 5):
    methods_selection = functools.reduce(operator.or_, (df['method'].str.startswith(method) for method in methods))

    display_df(df.loc[(methods_selection) & (df['num_flows'] == num_flows) & (
            df['end_epoch'].astype(int) == 99) & (df['beta'] == 1) & (df['amortized_flow'] == 0) & (
                              df['class_dim'] == 256) & (df['weighted_mixture'] == 0) & (
                              df['initial_learning_rate'].astype(float) == 0.0005)][[*show_cols]].sort_values(
        by=['score'], ascending=False))


if __name__ == '__main__':
    # experiment_uid = 'Mimic_joint_elbo_2021_07_06_09_44_52_871882'
    # experiment_uid = 'Mimic_mopgfm_2021_08_05_10_24_13_857815'
    # cond_gen(_id=experiment_uid, save_path='')
    # show_generated_figs(_id=experiment_uid)
    # experiments_database = MongoDatabase(training=False, _id=experiment_uid)
    # experiment_dict = experiments_database.get_experiment_dict()
    # plot_basic_batch_logs('train', experiment_dict)
    # plot_basic_batch_logs('test', experiment_dict)
    # plot_betas(experiment_dict)
    # plot_lr_accuracy(experiment_dict)
    # df = make_experiments_dataframe(experiments_database.connect())
    # plot_prd_scores(pd.DataFrame(df.loc[df['_id'] == 'polymnist_pgfm_2021_07_09_21_52_29_311887']))
    # compare_methods(df, methods=['gfm', 'joint_elbo'], df_selectors={'end_epoch': 99})
    for id in ['Mimic_mopgfm_2021_08_05_10_24_13_857815']:
        upload_notebook_to_db(id)
