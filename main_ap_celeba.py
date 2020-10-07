import sys
import os
import json

from celeba.experiment import LABELS

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-colorblind');
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


SUBSETS = ['img', 'text', 'img_text']
MODS = ['img', 'text'];


def create_bar_plot_all_subsets(fn_plot, values, b_w=0.3):
    index = np.arange(len(LABELS))
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    for m, s_key in enumerate(values.keys()):
        ax.bar(index+(m-1)*b_w, values[s_key]['values'], b_w, color=COLORS[m], alpha=0.5)
        ax.axhline(np.mean(np.fromiter(values[s_key]['values'], dtype=float)), 1,
                   len(LABELS), color=COLORS[m])
    ax.set_xticklabels(LABELS, rotation=90)
    ax.set_xticks(index)
    ax.set_ylim([0.0, 1.05])
    #ax.set_title(mod + ' ' + eval_type)
    plt.tight_layout()
    plt.legend(values.keys())
    plt.draw()
    plt.savefig(fn_plot, format='png')
    plt.close(fig)


def create_plot_representation(dir_results, dir_plot, str_exp):
    fn_rep = os.path.join(dir_results, str_exp + '_representation.csv');
    df_rep = pd.read_csv(fn_rep);
    val = dict();
    for k, key in enumerate(SUBSETS):
        rep = df_rep.loc[df_rep['subset'] == key];
        rep = rep.drop('subset', axis=1);
        print(rep.shape)
        rep_dict = {'labels': rep['label'].values,
                    'values': rep['value'].values}
        val[key] = rep_dict;
        mean_val = np.mean(rep['value'].dropna().values)
        print('mean ' + key + ': ' + str(mean_val))
    fn_plot_rep = os.path.join(dir_plot, 'rep.png');
    create_bar_plot_all_subsets(fn_plot_rep, val, b_w=0.3);


def create_plots_generation(dir_results, dir_plot, str_exp):
    fn_gen = os.path.join(dir_results, str_exp + '_generation.csv');
    df_gen = pd.read_csv(fn_gen);
    for k, in_key in enumerate(SUBSETS):
        gen = df_gen.loc[df_gen['subset'] == in_key];
        gen = gen.drop('subset', axis=1);
        val = dict();
        for l, out_key in enumerate(MODS):
            gen_out = gen.loc[gen['out'] == out_key];
            gen_out = gen_out.drop('out', axis=1);
            print(gen_out.shape)
            gen_dict = {'labels': gen_out['label'].values,
                        'values': gen_out['value'].values}
            val[out_key] = gen_dict;
            mean_val = np.mean(gen_out['value'].values)
            print('mean ' + in_key + ' -> ' + out_key + ': ' + str(mean_val))
        fn_plot_gen = os.path.join(dir_plot, 'gen_' + in_key + '_.png');
        create_bar_plot_all_subsets(fn_plot_gen, val, b_w=0.3);


if __name__ == '__main__':
    dir_res = sys.argv[1];
    exp = sys.argv[2];
    dir_plt = dir_res;

    create_plot_representation(dir_res, dir_plt, exp);
    create_plots_generation(dir_res, dir_plt, exp);

