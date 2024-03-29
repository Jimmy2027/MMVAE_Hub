import math

import numpy as np
import torch.nn.functional
from torch.utils.data import DataLoader

from mmvae_hub.utils.Dataclasses.Dataclasses import *
from mmvae_hub.utils.metrics.likelihood import log_marginal_estimate, log_joint_estimate
from mmvae_hub.utils.utils import dict_to_device

LOG2PI = float(np.log(2.0 * math.pi))


# at the moment: only marginals and joint
def calc_log_likelihood_batch(exp, latents: JointLatents, subset_key, subset, batch, num_imp_samples=10):
    flags = exp.flags
    model = exp.mm_vae
    mod_weights = exp.style_weights
    mods = exp.modalities

    n_total_samples = batch[list(batch)[0]].shape[0] * num_imp_samples

    if flags.factorized_representation:
        enc_mods = latents['modalities']
        style = model.get_random_style_dists(flags.batch_size)
        for mod in (subset):
            if (enc_mods[mod.name + '_style'][0] is not None
                    and enc_mods[mod.name + '_style'][1] is not None):
                style[mod.name] = enc_mods[mod.name + '_style']
    else:
        style = None

    mod_names = mods.keys()
    # sample from the subset posterior
    l = latents.get_latent_samples(subset_key=subset_key, n_imp_samples=num_imp_samples, mod_names=mod_names,
                                   style=style, model=model)

    l_style_rep = l['style']
    l_content_rep = l['content']

    c = {'mu': l_content_rep['mu'].view(n_total_samples, -1),
         'logvar': l_content_rep['logvar'].view(n_total_samples, -1),
         'z': l_content_rep['z'].view(n_total_samples, -1)}
    l_lin_rep = {'content': c, 'style': {}}
    for m_key in (l_style_rep.keys()):
        if flags.factorized_representation:
            s = {'mu': l_style_rep[mod.name]['mu'].view(n_total_samples, -1),
                 'logvar': l_style_rep[mod.name]['logvar'].view(n_total_samples, -1),
                 'z': l_style_rep[mod.name]['z'].view(n_total_samples, -1)}
            l_lin_rep['style'][m_key] = s
        else:
            l_lin_rep['style'][m_key] = None

    l_dec = ReparamLatent(content=l_lin_rep['content']['z'], style={})

    for m_key in (l_style_rep.keys()):
        if flags.factorized_representation:
            l_dec.style[m_key] = l_lin_rep['style'][m_key]['z']
        else:
            l_dec.style[m_key] = None

    gen = model.generate_sufficient_statistics_from_latents(l_dec)

    l_lin_rep_style = l_lin_rep['style']
    l_lin_rep_content = l_lin_rep['content']
    ll = {}

    for m_key, mod in mods.items():
        # compute marginal log-likelihood
        style_mod = l_lin_rep_style[mod.name] if mod in subset else None

        ll_mod = log_marginal_estimate(flags,
                                       num_imp_samples,
                                       gen[mod.name],
                                       mod.batch_text_to_onehot(batch[mod.name], mod.num_features)
                                       if mod.name == 'text'
                                       else batch[mod.name],
                                       style_mod,
                                       l_lin_rep_content)
        ll[mod.name] = ll_mod

    ll_joint = log_joint_estimate(mods, flags, num_imp_samples, gen, batch,
                                  l_lin_rep_style,
                                  l_lin_rep_content)
    ll['joint'] = ll_joint
    return ll


def estimate_likelihoods(exp):
    model = exp.mm_vae
    mods = exp.modalities
    bs_normal = exp.flags.batch_size
    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=exp.flags.dataloader_workers, drop_last=True)

    subsets = exp.subsets
    if '' in subsets:
        del subsets['']
    lhoods = {}
    for s_key in subsets:
        lhoods[s_key] = {m_key: [] for m_key in mods}
        lhoods[s_key]['joint'] = []

    for batch in d_loader:
        batch_d = dict_to_device(batch[0], exp.flags.device)

        _, joint_latent = model.inference(batch_d)
        for s_key in (subsets.keys()):
            subset = subsets[s_key]
            ll_batch = calc_log_likelihood_batch(exp, joint_latent,
                                                 s_key, subset,
                                                 batch_d,
                                                 num_imp_samples=6)
            for m_key in (ll_batch.keys()):
                lhoods[s_key][m_key].append(ll_batch[m_key].item())

    for s_key, lh_subset in lhoods.items():
        for m_key in (lh_subset.keys()):
            mean_val = np.mean(np.array(lh_subset[m_key]))
            lhoods[s_key][m_key] = mean_val
    exp.flags.batch_size = bs_normal
    return lhoods
