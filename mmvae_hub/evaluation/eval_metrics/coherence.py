import typing
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmvae_hub import log
from mmvae_hub.utils.plotting.save_samples import save_generated_samples_singlegroup
from mmvae_hub.utils.utils import init_twolevel_nested_dict, dict_to_device


def classify_cond_gen_samples(exp, labels: Tensor, cond_samples: typing.Mapping[str, Tensor]) \
        -> typing.Mapping[str, Tensor]:
    """
    Classifies for each modality all the conditionally generated samples.
    Returns a dict like the following:
    {'PA': tensor,
    'Lateral': tensor,
    'text': tensor}

    """
    clf_predictions = {mod: {} for mod in exp.modalities}
    for mod in exp.modalities:
        if mod in cond_samples:
            mod_cond_gen: Tensor = cond_samples[mod]
            mod_clf = exp.modalities[mod].clf
            # classify generated sample to evaluate coherence
            clf_predictions[mod] = mod_clf(mod_cond_gen).cpu()
        else:
            log.info(str(mod) + 'not existing in cond_gen_samples')
    return clf_predictions


def calculate_coherence(exp, samples) -> dict:
    """
    Classifies generated modalities. The generated samples are coherent if all modalities
    are classified as belonging to the same class.
    """
    mods = exp.modalities
    # TODO: make work for num samples NOT EQUAL to batch_size
    c_labels = {}
    for j, l_key in enumerate(exp.labels):
        pred_mods = np.zeros((len(mods.keys()), exp.flags.batch_size))
        for k, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            clf_mod = mod.clf
            samples_mod = samples[mod.name]
            attr_mod = clf_mod(samples_mod)
            output_prob_mod = attr_mod.cpu().data.numpy()
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int)
            pred_mods[k, :] = pred_mod
        coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
        coherence = np.sum(coh_mods.astype(int)) / float(exp.flags.batch_size)
        c_labels[l_key] = coherence
    return c_labels


def transform_gen_samples(gen_samples, transform):
    """
    transforms the generated samples as needed for the classifier
    """

    transformed_samples = [
        transform(gen_samples[idx].cpu())
        for idx in range(gen_samples.shape[0])
    ]

    return torch.stack(transformed_samples)


def save_generated_samples(exp, rand_gen: dict, iteration: int, batch_d: dict) -> None:
    """
    Saves generated samples to dir_fid
    """
    save_generated_samples_singlegroup(exp, iteration, 'random', rand_gen)
    if exp.flags.__contains__('text_encoding') and exp.flags.text_encoding == 'word':
        batch_d_temp = batch_d.copy()
        batch_d_temp['text'] = torch.nn.functional.one_hot(batch_d_temp['text'].to(torch.int64),
                                                           num_classes=exp.flags.vocab_size)

        save_generated_samples_singlegroup(exp, iteration,
                                           'real',
                                           batch_d_temp)
    else:
        save_generated_samples_singlegroup(exp, iteration,
                                           'real',
                                           batch_d)


def init_gen_perf(labels, subsets, mods) -> typing.Mapping[str, dict]:
    """
    Initialises gen_perf dict with empty iterables.
    The result will look like this:

    {'cond':
        {'Lung Opacity':
            {'PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()}},
      'Pleural Effusion':
          {'PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()}},
      'Support Devices':
          {'PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()},
           'Lateral_PA_text': {'PA': Tensor(), 'Lateral': Tensor(), 'text': Tensor()}}},

    'random':
        {'Lung Opacity': Tensor(), 'Pleural Effusion': Tensor(), 'Support Devices': Tensor()}}
    """
    return {'cond': init_twolevel_nested_dict(labels, subsets, init_val={mod: torch.Tensor() for mod in mods}),
            'random': {k: [].copy() for k in labels}}


def calc_coherence_random_gen(exp, mm_vae, iteration: int, rand_coherences: Mapping[str, typing.List], batch_d: dict) -> \
        Mapping[str, typing.List]:
    args = exp.flags
    # generating random samples
    rand_gen = mm_vae.module.generate() if args.distributed else mm_vae.generate()
    rand_gen = dict_to_device(rand_gen, args.device)
    # classifying generated examples
    coherence_random = calculate_coherence(exp, rand_gen)
    for j, l_key in enumerate(exp.labels):
        rand_coherences[l_key].append(coherence_random[l_key])

    if (exp.flags.batch_size * iteration) < exp.flags.num_samples_fid and args.save_figure:
        # saving generated samples to dir_fid
        save_generated_samples(exp, rand_gen, iteration, batch_d)

    return rand_coherences


def eval_classified_gen_samples(exp, subsets, mods, cond_gen_classified, rand_coherences, batch_labels):
    """
    Computes the eval metric of the predicted classes.
    """
    gen_perf = {'random': {}}
    gen_perf_cond = {}
    # compare the classification on the generated samples with the ground truth
    for l_idx, l_key in enumerate(exp.labels):
        gen_perf_cond[l_key] = {}
        for s_key in subsets:
            gen_perf_cond[l_key][s_key] = {}
            for m_key in mods:
                perf = exp.eval_label(cond_gen_classified[s_key][m_key].cpu().data.numpy(), batch_labels, l_idx)
                gen_perf_cond[l_key][s_key][m_key] = perf

        eval_score = exp.mean_eval_metric(rand_coherences[l_key])
        gen_perf['random'][l_key] = eval_score

    gen_perf['cond'] = gen_perf_cond

    return gen_perf


def test_generation(exp, dataset=None):
    """
    Generates random and conditioned samples and evaluates coherence.
    """
    args = exp.flags

    mods = exp.modalities
    mm_vae = exp.mm_vae
    subsets = [*exp.subsets, 'joint']

    d_loader = DataLoader(exp.dataset_test if not dataset else dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True)

    batch_labels, rand_coherences, cond_gen_classified = classify_generated_samples(args, d_loader, exp,
                                                                                    mm_vae,
                                                                                    mods, subsets)

    return eval_classified_gen_samples(exp, subsets, mods, cond_gen_classified, rand_coherences, batch_labels)


def classify_generated_samples(args, d_loader, exp, mm_vae, mods, subsets):
    """
    Generates and classifies samples.
    """

    labels = exp.labels
    rand_coherences = {k: [].copy() for k in labels}

    # all labels accumulated over batches:
    batch_labels = torch.Tensor()
    cond_gen_classified = init_twolevel_nested_dict(subsets, mods, init_val=torch.Tensor())
    cond_gen_classified: Mapping[str, Mapping[mods, Tensor]]

    # for iteration, (batch_d, batch_l) in enumerate(d_loader):
    for iteration, (batch_d, batch_l) in tqdm(enumerate(d_loader), total=len(d_loader)):

        batch_labels = torch.cat((batch_labels, batch_l), 0)
        batch_d = dict_to_device(batch_d, exp.flags.device)

        # evaluating random generation
        rand_coherences = calc_coherence_random_gen(exp, mm_vae, iteration, rand_coherences, batch_d)

        # evaluating conditional generation
        # first generates the conditional gen_samples
        # classifies them and stores the classifier predictions
        _, joint_latent = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)

        cg = mm_vae.module.cond_generation(joint_latent) if args.distributed else mm_vae.cond_generation(joint_latent)
        cg: typing.Mapping[str, typing.Mapping[mods, Tensor]]
        # classify the cond. generated samples
        for subset, cond_val in cg.items():
            clf_cg: Mapping[mods, Tensor] = classify_cond_gen_samples(exp, batch_l, cond_val)
            for mod in mods:
                cond_gen_classified[subset][mod] = torch.cat((cond_gen_classified[subset][mod], clf_cg[mod]), 0)
            if (exp.flags.batch_size * iteration) < exp.flags.num_samples_fid and exp.flags.save_figure:
                save_generated_samples_singlegroup(exp, iteration, subset, cond_val)

    return batch_labels, rand_coherences, cond_gen_classified


def flatten_cond_gen_values(gen_eval: dict) -> dict:
    """
    Converts the coherence evaluation results into a flattened dict
    """
    flattened_dict = {}
    for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
        for k, s_key in enumerate(gen_eval['cond'][l_key].keys()):
            for g_key in gen_eval['cond'][l_key][s_key]:
                key = l_key + '_' + s_key + '__' + g_key
                flattened_dict[key] = gen_eval['cond'][l_key][s_key][g_key]

    flattened_dict['random'] = {k: v for k, v in gen_eval['random'].items()}
    return flattened_dict
