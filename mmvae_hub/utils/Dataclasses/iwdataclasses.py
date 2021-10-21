from typing import Tuple

from torch.distributions import Distribution

from mmvae_hub.utils.Dataclasses.Dataclasses import *
from dataclasses import dataclass
from torch import Tensor

from mmvae_hub.utils.Dataclasses.gfmDataclasses import JointLatentsGfM


@dataclass
class JointLatentsiwMoGfM2:
    """Joint Latens for generalized f-means methods."""
    joint_embedding: JointEmbeddingFoEM
    z_Gfs: Mapping[str, Tensor]
    subset_samples: Mapping[str, Tuple]
    subsets: Mapping[str, Tensor]
    enc_mods: Mapping[str, BaseEncMod]
    srv_proxies: Mapping[str,Distribution]
    def get_joint_embeddings(self):
        return self.joint_embedding.embedding.mean(dim=0)

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key][0].mean(dim=0)

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding.mean(dim=0)
        return self.subsets[subset_key][0].mean(dim=0)

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key][0].mean(dim=0).cpu()

        lr_data['q0']['joint'] = self.joint_embedding.embedding.mean(dim=0).cpu()

        return lr_data

    def get_latent_samples(self, subset_key: str, n_imp_samples, model, mod_names=None, style=None):
        """Sample n_imp_samples from the latents."""
        # modalities that span the subset
        enc_mod_selection = subset_key.split('_')

        batch_size, class_dim = self.enc_mods[[mod_str for mod_str in self.enc_mods][0]].latents_class.mu.shape

        transformed_enc_mods = {
            mod_key: model.flow(
                torch.cat([distr.latents_class.reparameterize().unsqueeze(dim=0) for _ in range(n_imp_samples)],
                          dim=0).reshape((n_imp_samples * batch_size, class_dim)))[
                0] for mod_key, distr in
            self.enc_mods.items() if mod_key in enc_mod_selection}

        subset_zks = torch.Tensor().to(self.enc_mods[[mod_str for mod_str in self.enc_mods][0]].latents_class.mu.device)
        for mod_k in enc_mod_selection:
            subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod_k].unsqueeze(dim=0)), dim=0)
        # mean of zks
        z_mean = torch.mean(subset_zks, dim=0)
        # calculate inverse flow
        samples = model.flow.rev(z_mean)[0].reshape((n_imp_samples, batch_size, class_dim))

        c = {'mu': None, 'logvar': None, 'z': samples}

        return {'content': c, 'style': None}


@dataclass
class JointLatentsiwMoGfMVAE_amortized:
    """Joint Latens for generalized f-means methods."""
    joint_embedding: JointEmbeddingFoEM
    transformed_enc_mods: Mapping[str, Tuple]
    subset_samples: Mapping[str, Tuple]
    subsets: Mapping[str, Tensor]
    enc_mods: Mapping[str, BaseEncMod]
    zmss: Mapping[str, Tensor]
    priors_tf_enc_mods: Mapping[str, Distribution]

    def get_joint_embeddings(self):
        return self.joint_embedding.embedding.mean(dim=0)

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key][0].mean(dim=0)

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding.mean(dim=0)
        return self.subsets[subset_key][0].mean(dim=0)

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key][0].mean(dim=0).cpu()

        lr_data['q0']['joint'] = self.joint_embedding.embedding.mean(dim=0).cpu()

        return lr_data

    def get_latent_samples(self, subset_key: str, n_imp_samples, model, mod_names=None, style=None):
        """Sample n_imp_samples from the latents."""
        # modalities that span the subset
        enc_mod_selection = subset_key.split('_')

        batch_size, class_dim = self.enc_mods[[mod_str for mod_str in self.enc_mods][0]].latents_class.mu.shape

        transformed_enc_mods = {
            mod_key: model.flow(
                torch.cat([distr.latents_class.reparameterize().unsqueeze(dim=0) for _ in range(n_imp_samples)],
                          dim=0).reshape((n_imp_samples * batch_size, class_dim)))[
                0] for mod_key, distr in
            self.enc_mods.items() if mod_key in enc_mod_selection}

        subset_zks = torch.Tensor().to(self.enc_mods[[mod_str for mod_str in self.enc_mods][0]].latents_class.mu.device)
        for mod_k in enc_mod_selection:
            subset_zks = torch.cat((subset_zks, transformed_enc_mods[mod_k].unsqueeze(dim=0)), dim=0)
        # mean of zks
        z_mean = torch.mean(subset_zks, dim=0)
        # calculate inverse flow
        samples = model.flow.rev(z_mean)[0].reshape((n_imp_samples, batch_size, class_dim))

        c = {'mu': None, 'logvar': None, 'z': samples}

        return {'content': c, 'style': None}


@dataclass
class JointLatentsiwMoGfM(JointLatentsGfM):
    """Joint Latents for mixture of generalized f-means methods."""
    epss: Tensor

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].mean(dim=0).cpu()

        lr_data['q0']['joint'] = self.joint_embedding.embedding.mean(dim=0).cpu()

        return lr_data

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].mean(dim=0)

    def get_joint_embeddings(self):
        return self.joint_embedding.embedding.mean(dim=0)


@dataclass
class iwSubset:
    zs: Tensor
    qz_x_tilde: Distribution


@dataclass
class iwJointLatents:
    fusion_subsets_keys: Iterable[str]
    joint_distr: iwSubset
    subsets: Mapping[str, iwSubset]
    zss: Mapping[str, Distribution]

    def get_joint_embeddings(self):
        return self.joint_distr.qz_x_tilde.rsample()

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].qz_x_tilde.rsample()

    def get_q0(self, subset_key: str):
        """Return the mean of the subset."""
        if subset_key == 'joint':
            return self.get_joint_q0()
        return self.subsets[subset_key].qz_x_tilde.mean

    def get_joint_q0(self):
        return self.joint_distr.qz_x_tilde.mean

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
        c_embed = self.subsets[subset_key].qz_x_tilde.rsample((n_imp_samples,))

        c = {'mu': self.subsets[subset_key].qz_x_tilde.loc.unsqueeze(0).repeat(n_imp_samples, 1, 1),
             'logvar': self.subsets[subset_key].qz_x_tilde.scale.unsqueeze(0).repeat(n_imp_samples, 1, 1),
             'z': c_embed}

        styles = {key: None for k, key in enumerate(mod_names)}
        return {'content': c, 'style': styles}

    # def get_latent_samples(self, subset_key: str, n_imp_samples, mod_names=None, style=None, model=None):
    #     """Sample n_imp_samples from the latents."""
    #     c_embed = self.subsets[subset_key].qz_x_tilde.rsample(n_imp_samples)
    #     l_s = style
    #     l_c_m_rep = l_c.mu.unsqueeze(0).repeat(n_imp_samples, 1, 1)
    #     l_c_lv_rep = l_c.logvar.unsqueeze(0).repeat(n_imp_samples, 1, 1)
    #     c_emb = Distr(l_c_m_rep, l_c_lv_rep).reparameterize()
    #
    #     styles = {}
    #     c = {'mu': l_c_m_rep, 'logvar': l_c_lv_rep, 'z': c_emb}
    #
    #     if style is not None:
    #         for k, key in enumerate(l_s.keys()):
    #             l_s_mod = l_s[key]
    #             l_s_m_rep = l_s_mod[0].unsqueeze(0).repeat(n_imp_samples, 1, 1)
    #             l_s_lv_rep = l_s_mod[1].unsqueeze(0).repeat(n_imp_samples, 1, 1)
    #             s_emb = Distr(l_s_m_rep, l_s_lv_rep).reparameterize()
    #             s = {'mu': l_s_m_rep, 'logvar': l_s_lv_rep, 'z': s_emb}
    #             styles[key] = s
    #     else:
    #         for k, key in enumerate(mod_names):
    #             styles[key] = None
    #
    #     return {'content': c, 'style': styles}


@dataclass
class iwForwardResults:
    enc_mods: Mapping[str, BaseEncMod]
    joint_latents: iwJointLatents
    rec_mods: dict
