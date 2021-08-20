from torch.distributions import Distribution

from mmvae_hub.utils.dataclasses.Dataclasses import *
from dataclasses import dataclass
import torch.distributions as distr
from torch import Tensor


@dataclass
class iwSubset:
    zs: Tensor
    qz_x_tilde: Distribution


@dataclass
class iwJointLatents:
    fusion_subsets_keys: Iterable[str]
    subsets: Mapping[str, iwSubset]
    zss: Mapping[str, Distribution]

    def get_joint_embeddings(self):
        return self.joint_distr.reparameterize()

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].reparameterize()

    def get_q0(self, subset_key: str):
        """Return the mean of the subset."""
        if subset_key == 'joint':
            return self.get_joint_q0()
        return self.subsets[subset_key].mu

    def get_joint_q0(self):
        return self.joint_distr.mu

    def get_lreval_data(self) -> dict:
        """Get lr_data for the lr evaluation."""
        lr_data = {'q0': {}}
        for key in self.subsets:
            lr_data['q0'][key] = self.get_q0(key).cpu()

        lr_data['q0']['joint'] = self.get_joint_q0().cpu()

        return lr_data

    def get_lreval_data_(self, data_train: dict):
        """Add lr values to data_train."""
        for key in self.subsets:
            data_train['q0'][key] = torch.cat((data_train['q0'][key], self.get_q0(key).cpu()), 0)
        joint_q0 = self.get_joint_q0().cpu()
        data_train['q0']['joint'] = torch.cat((data_train['q0']['joint'], joint_q0), 0)

        return data_train

    def get_latent_samples(self, subset_key: str, n_imp_samples, mod_names=None, style=None, model=None):
        """Sample n_imp_samples from the latents."""
        l_c = self.subsets[subset_key]
        l_s = style
        l_c_m_rep = l_c.mu.unsqueeze(0).repeat(n_imp_samples, 1, 1)
        l_c_lv_rep = l_c.logvar.unsqueeze(0).repeat(n_imp_samples, 1, 1)
        c_emb = Distr(l_c_m_rep, l_c_lv_rep).reparameterize()

        styles = {}
        c = {'mu': l_c_m_rep, 'logvar': l_c_lv_rep, 'z': c_emb}

        if style is not None:
            for k, key in enumerate(l_s.keys()):
                l_s_mod = l_s[key]
                l_s_m_rep = l_s_mod[0].unsqueeze(0).repeat(n_imp_samples, 1, 1)
                l_s_lv_rep = l_s_mod[1].unsqueeze(0).repeat(n_imp_samples, 1, 1)
                s_emb = Distr(l_s_m_rep, l_s_lv_rep).reparameterize()
                s = {'mu': l_s_m_rep, 'logvar': l_s_lv_rep, 'z': s_emb}
                styles[key] = s
        else:
            for k, key in enumerate(mod_names):
                styles[key] = None

        return {'content': c, 'style': styles}


@dataclass
class iwForwardResults:
    enc_mods: Mapping[str, BaseEncMod]
    joint_latents: iwJointLatents
    rec_mods: dict
