import math

import torch

from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.utils import reweight_weights


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    # print('logvar: ', log_var.mean().item())
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    # print('log_norm_diag: ', log_norm.mean().item())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x
    # print('log_norm: ', log_norm.mean().item())

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def calc_kl_divergence_flow(distr0: Distr = None, distr1: Distr = None, enc_mod: EncModPlanarMixture = None,
                            norm_value=None) -> Tensor:
    """
    Calculate the KL Divergence: DKL = E_q0[ ln q(z_0) - ln p(z_k) ] - E_q_z0[\sum_k log |det dz_k/dz_k-1|].
    """

    # get the mean and variance of z0
    mu0, logvar0 = enc_mod.latents_class.mu, enc_mod.latents_class.logvar

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(enc_mod.zk, dim=1)
    # print('z0: ', (enc_mod.z0.mean().item(), enc_mod.z0.std().item()), 'z_k: ', (enc_mod.zk.mean().item(), enc_mod.zk.std().item()))
    # ln q(z_0)  (not averaged)
    # fixme logvar (et du coup z0) gets huge and log_q_z0 contains nans.
    log_q_z0 = log_normal_diag(x=enc_mod.z0, mean=mu0, log_var=logvar0, dim=1)
    # print('log_q_z0', log_q_z0.mean().item(), 'log_p_zk: ', log_p_zk.mean().item())
    # log_q_z0 = torch.nan_to_num(log_q_z0)

    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    diff = log_q_z0 - log_p_zk
    # to minimize the divergence,
    summed_logs = torch.sum(diff.abs())

    # sum over batches
    summed_ldj = torch.sum(enc_mod.log_det_j)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    KLD = (summed_logs - summed_ldj)
    # print('summed_ldj: ', summed_ldj)
    # print('KLD: ', KLD, '\n')

    if norm_value is not None:
        KLD = KLD / float(norm_value)

    # assert KLD.cpu().item() >= 0

    return KLD


def calc_kl_divergence(distr0: Distr, distr1: Distr = None, enc_mod: BaseEncMod = None,
                       norm_value: int = None) -> Tensor:
    mu0, logvar0 = distr0.mu, distr0.logvar

    if distr1 is None:
        KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)

    else:
        mu1, logvar1 = distr1.mu, distr1.logvar
        KLD = -0.5 * (
            torch.sum(1 - logvar0.exp() / logvar1.exp() - (mu0 - mu1).pow(2) / logvar1.exp() + logvar0 - logvar1))

    if norm_value is not None:
        KLD = KLD / float(norm_value)

    return KLD


def calc_gaussian_scaling_factor(PI, mu1, logvar1, mu2=None, logvar2=None, norm_value=None):
    d = mu1.shape[1];
    if mu2 is None or logvar2 is None:
        # print('S_11: ' + str(torch.sum(1/((2*PI*(logvar1.exp() + 1)).pow(0.5)))))
        # print('S_12: ' + str(torch.sum(torch.exp(-0.5*(mu1.pow(2)/(logvar1.exp()+1))))))
        S_pre = (1 / (2 * PI).pow(d / 2)) * torch.sum((logvar1.exp() + 1), dim=1).pow(0.5);
        S = S_pre * torch.sum((-0.5 * (mu1.pow(2) / (logvar1.exp() + 1))).exp(), dim=1);
        S = torch.sum(S)
    else:
        # print('S_21: ' + str(torch.sum(1/((2*PI).pow(d/2)*(logvar1.exp()+logvar2.exp()).pow(0.5)))));
        # print('S_22: ' + str(torch.sum(torch.exp(-0.5 * ((mu1 - mu2).pow(2) / (logvar1.exp() + logvar2.exp()))))));
        S_pre = torch.sum(1 / ((2 * PI).pow(d / 2) * (logvar1.exp() + logvar2.exp())), dim=1).pow(0.5)
        S = S_pre * torch.sum(torch.exp(-0.5 * ((mu1 - mu2).pow(2) / (logvar1.exp() + logvar2.exp()))), dim=1);
        S = torch.sum(S)
    if norm_value is not None:
        S = S / float(norm_value);
    # print('S: ' + str(S))
    return S


def calc_gaussian_scaling_factor_self(PI, logvar1, norm_value=None):
    d = logvar1.shape[1];
    S = (1 / (2 * PI).pow(d / 2)) * torch.sum(logvar1.exp(), dim=1).pow(0.5);
    S = torch.sum(S);
    # S = torch.sum(1 / (2*(PI*torch.exp(logvar1)).pow(0.5)));
    if norm_value is not None:
        S = S / float(norm_value);
    # print('S self: ' + str(S))
    return S


# def calc_kl_divergence_lb_gauss_mixture(flags, index, mu1, logvar1, mus, logvars, norm_value=None):
#     klds = torch.zeros(mus.shape[0]+1)
#     if flags.cuda:
#         klds = klds.cuda();
#
#     klds[0] = calc_kl_divergence(mu1, logvar1, norm_value=norm_value);
#     for k in range(0, mus.shape[0]):
#         if k == index:
#             kld = 0.0;
#         else:
#             kld = calc_kl_divergence(mu1, logvar1, mus[k], logvars[k], norm_value=norm_value);
#         klds[k+1] = kld;
#     kld_mixture = klds.mean();
#     return kld_mixture;

def calc_kl_divergence_lb_gauss_mixture(flags, index, mu1, logvar1, mus, logvars, norm_value=None):
    PI = torch.Tensor([math.pi]);
    w_modalities = torch.Tensor(flags.alpha_modalities);
    if flags.cuda:
        PI = PI.cuda();
        w_modalities = w_modalities.cuda();
    w_modalities = reweight_weights(w_modalities);

    denom = w_modalities[0] * calc_gaussian_scaling_factor(PI, mu1, logvar1, norm_value=norm_value);
    for k in range(0, len(mus)):
        if index == k:
            denom += w_modalities[k + 1] * calc_gaussian_scaling_factor_self(PI, logvar1, norm_value=norm_value);
        else:
            denom += w_modalities[k + 1] * calc_gaussian_scaling_factor(PI, mu1, logvar1, mus[k], logvars[k],
                                                                        norm_value=norm_value)
    lb = -torch.log(denom);
    return lb;


def calc_kl_divergence_ub_gauss_mixture(flags, index, mu1, logvar1, mus, logvars, entropy, norm_value=None):
    PI = torch.Tensor([math.pi]);
    w_modalities = torch.Tensor(flags.alpha_modalities);
    if flags.cuda:
        PI = PI.cuda();
        w_modalities = w_modalities.cuda();
    w_modalities = reweight_weights(w_modalities);

    nom = calc_gaussian_scaling_factor_self(PI, logvar1, norm_value=norm_value);
    kl_div = calc_kl_divergence(mu1, logvar1, norm_value=norm_value);
    print('kl div uniform: ' + str(kl_div))
    denom = w_modalities[0] * torch.min(torch.Tensor([kl_div.exp(), 100000]));
    for k in range(0, len(mus)):
        if index == k:
            denom += w_modalities[k + 1];
        else:
            kl_div = calc_kl_divergence(mu1, logvar1, mus[k], logvars[k], norm_value=norm_value)
            print('kl div ' + str(k) + ': ' + str(kl_div))
            denom += w_modalities[k + 1] * torch.min(torch.Tensor([kl_div.exp(), 100000]));
    ub = torch.log(nom) - torch.log(denom) + entropy;
    return ub;


def calc_entropy_gauss(flags, logvar, norm_value=None):
    PI = torch.Tensor([math.pi]);
    if flags.cuda:
        PI = PI.cuda();
    ent = 0.5 * torch.sum(torch.log(2 * PI) + logvar + 1)
    if norm_value is not None:
        ent = ent / norm_value;
    return ent;


if __name__ == '__main__':
    a = torch.ones((256, 64)) * 0.001
    enc_mods = EncModPlanarMixture(z0=a, zk=a, log_det_j=torch.zeros((256, 64)), latents_class=Distr(logvar=a, mu=a),
                                   flow_params=0)
    calc_kl_divergence_flow(enc_mod=enc_mods)
