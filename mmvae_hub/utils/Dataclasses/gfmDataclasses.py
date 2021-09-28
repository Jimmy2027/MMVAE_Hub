from mmvae_hub.utils.Dataclasses.Dataclasses import *
from mmvae_hub.utils.Dataclasses.Dataclasses import JointEmbeddingFoEM, BaseEncMod, Distr, JointLatentsFoS


@dataclass
class JointLatentsGfM:
    """Joint Latens for generalized f-means methods."""
    joint_embedding: JointEmbeddingFoEM
    subset_samples: Mapping[str, Tensor]
    subsets: Mapping[str, Tensor]
    enc_mods: Mapping[str, BaseEncMod]

    def get_joint_embeddings(self):
        return self.joint_embedding.embedding

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key]

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding
        return self.subsets[subset_key]

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].cpu()

        lr_data['q0']['joint'] = self.subsets['_'.join(self.joint_embedding.mod_strs)].cpu()

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
class SubsetMoFoGfM:
    """Subset of Mixture of Flow of GfM"""
    z0: Optional[Tensor] = None
    zk: Optional[Tensor] = None
    log_det_j: Optional[Tensor] = None


@dataclass
class JointLatentsMoGfM(JointLatentsGfM):
    """Joint Latents for mixture of generalized f-means methods."""
    epss: Tensor

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].mean(dim=0).cpu()

        lr_data['q0']['joint'] = self.joint_embedding.embedding.mean(dim=0).cpu()

        return lr_data


@dataclass
class JointLatentsGfMoP(JointLatentsGfM):
    subsets: Mapping[str, Distr]

    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key].reparameterize()

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding
        return self.subsets[subset_key].mu

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].mu.cpu()

        lr_data['q0']['joint'] = self.get_joint_embeddings().cpu()

        return lr_data


@dataclass
class JointLatentsEGfM(JointLatentsGfM):
    def get_subset_embedding(self, s_key: str):
        return self.subsets[s_key]

    def get_q0(self, subset_key: str):
        """Get the mean of the unimodal latents and the embeddings of the multimodal latents."""
        if subset_key == 'joint':
            return self.joint_embedding.embedding
        return self.subsets[subset_key]

    def get_lreval_data(self):
        lr_data = {'q0': {}}

        for key in self.subsets:
            lr_data['q0'][key] = self.subsets[key].cpu()

        lr_data['q0']['joint'] = self.subsets['_'.join(self.joint_embedding.mod_strs)].cpu()

        return lr_data


@dataclass
class JointLatentsMoFoGfM(JointLatentsFoS):
    """Joint Latens for Mixture of Flow of generalized f-means methods."""
    joint_embedding: JointEmbeddingFoEM
    subsets: Mapping[str, SubsetMoFoGfM]

    def get_z0(self, subset_key: str):
        """Return the mean of q0."""
        if subset_key == 'joint':
            return self.get_joint_q0()
        else:
            return self.subsets[subset_key].z0

    def get_joint_z0(self):
        """Return joint z0."""
        return self.joint_embedding.embedding

    def get_lreval_data(self):
        lr_data = {'q0': {}, 'zk': {}}
        for key in self.subsets:
            lr_data['q0'][key] = self.get_z0(key).cpu()

        lr_data['q0']['joint'] = self.get_joint_z0().cpu()

        for key in self.subsets:
            # get the latents after application of flows.
            lr_data['zk'][key] = self.get_zk(key).cpu()

        return lr_data
