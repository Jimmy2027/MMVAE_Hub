import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal

from mmvae_hub.experiment_vis.utils import load_experiment

exp_uid = 'polymnist_iwmogfm_multiloss_2021_09_28_11_08_24_742311'
exp = load_experiment(_id=exp_uid)

exp.set_eval_mode()
num_samples = 10
Gf = Normal(torch.zeros(exp.flags.class_dim, device=exp.flags.device),
            torch.tensor(1 / 2).sqrt() * torch.ones(exp.flags.class_dim, device=exp.flags.device))
z_Gf = Gf.sample_n(num_samples)

zss, log_det_J = exp.mm_vae.flow.rev(z_Gf)

rec_mods = {}
for mod_key, mod in exp.mm_vae.modalities.items():
    rec_mods[mod_key] = mod.calc_likelihood(None, class_embeddings=zss).mean

for k in range(num_samples):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(rec_mods['m0'][k].detach().cpu().numpy().swapaxes(0, -1))
    plt.subplot(1, 3, 2)
    plt.imshow(rec_mods['m1'][k].detach().cpu().numpy().swapaxes(0, -1))
    plt.subplot(1, 3, 3)
    plt.imshow(rec_mods['m2'][k].detach().cpu().numpy().swapaxes(0, -1))
    plt.show()
    plt.close()


