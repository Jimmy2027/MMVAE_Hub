from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch
from torch import Tensor


@dataclass
class VQEncMod:
    enc_mod: Tensor


@dataclass
class QuantizedLatent:
    quantizer_loss: Tensor
    quantized: Tensor
    perplexity: Tensor


@dataclass
class JointEmbeddingVQ:
    embedding: Tensor
    mod_strs: Iterable[str]


@dataclass
class JointLatentsVQ:
    """Joint Latens for generalized f-means methods."""
    joint_embedding: JointEmbeddingVQ
    subsets: Mapping[str, Tensor]

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
class VQForwardResults:
    enc_mods: Mapping[str, VQEncMod]
    quantized_latents: Mapping[str, QuantizedLatent]
    rec_mods: Mapping[str, Mapping[str, Tensor]]


@dataclass
class VQBatchResults:
    total_loss: Tensor
    quant_losses: dict
    rec_losses: dict


@dataclass
class VQTestResults(VQBatchResults):
    prd_scores: Optional[dict] = None
    lr_eval: Optional[dict] = None
    gen_eval: Optional[dict] = None
    end_epoch: Optional[int] = None
    mean_epoch_time: Optional[float] = None
    experiment_duration: Optional[float] = None
    hyperopt_score: Optional[int] = None
