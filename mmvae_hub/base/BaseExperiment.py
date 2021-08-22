import os
import random
import typing
from abc import abstractmethod, ABC
from itertools import chain, combinations
from typing import Mapping, Iterable

import numpy as np
import torch
from torch import optim, Tensor

from mmvae_hub.modalities import BaseModality
from mmvae_hub.networks.FlowVaes import PlanarMixtureMMVae, PfomMMVAE, PoPE, FoMFoP, FoMoP, AfomMMVAE, \
    MoFoPoE
from mmvae_hub.networks.GfMVaes import GfMVAE, GfMoPVAE, PGfMVAE, MopGfM, MoGfMVAE, MoFoGfMVAE, BMoGfMVAE, MoGfMVAE_old, \
    iwMoGfMVAE
from mmvae_hub.networks.MixtureVaes import MOEMMVae, MoPoEMMVae, JSDMMVae
from mmvae_hub.networks.PoEMMVAE import POEMMVae
from mmvae_hub.networks.iwVaes import iwMoE, iwMoPoE
from mmvae_hub.sylvester_flows.models.VAE import PlanarVAE, VAE
from mmvae_hub.utils import utils
from mmvae_hub.utils.MongoDB import MongoDatabase


class BaseExperiment(ABC):
    def __init__(self, flags):
        self.set_random_seed(hasattr(flags, 'deterministic') and flags.deterministic, flags.seed)
        self.flags = flags
        self.name = flags.dataset

        self.modalities = None
        self.num_modalities = None
        self.subsets = None
        self.dataset_train = None
        self.dataset_test = None

        self.mm_vae = None
        self.optimizer = None
        self.rec_weights = None
        self.style_weights = None

        self.test_samples = None
        self.paths_fid = None
        self.plot_img_size = None

        if flags.use_db:
            try:
                self.experiments_database = MongoDatabase(flags)
            except:
                self.flags.use_db = 2

    def set_model(self):
        """Chose the right VAE model depending on the chosen method."""
        if self.flags.method == 'mopoe':
            model = MoPoEMMVae(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'moe':
            model = MOEMMVae(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'poe':
            model = POEMMVae(self, flags=self.flags, modalities=self.modalities, subsets=self.subsets)
        elif self.flags.method == 'planar_mixture':
            model = PlanarMixtureMMVae(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'jsd':
            model = JSDMMVae(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'pfom':
            model = PfomMMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'pope':
            if self.flags.amortized_flow:
                raise NotImplementedError(f'Amortized flows are not implemented for the PoPE method.')
            model = PoPE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'fomfop':
            if self.flags.amortized_flow:
                raise NotImplementedError(f'Amortized flows are not implemented for the fomfop method.')
            model = FoMFoP(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'fomop':
            if self.flags.amortized_flow:
                raise NotImplementedError(f'Amortized flows are not implemented for the fomop method.')
            model = FoMoP(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'pgfm':
            model = PGfMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'gfm':
            model = GfMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'afom':
            model = AfomMMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'gfmop':
            model = GfMoPVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'mofop':
            model = MoFoPoE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'pgfmop':
            model = MoFoPoE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'planar_vae':
            model = PlanarVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'sylvester_vae_noflow':
            model = VAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'mopgfm':
            model = MopGfM(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'mogfm':
            model = MoGfMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'mofogfm':
            if self.flags.num_flows == 0:
                model = MoGfMVAE(self, self.flags, self.modalities, self.subsets)
            else:
                model = MoFoGfMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'bmogfm':
            model = BMoGfMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'mogfm_old':
            model = MoGfMVAE_old(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'iwmogfm':
            model = iwMoGfMVAE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'iwmoe':
            model = iwMoE(self, self.flags, self.modalities, self.subsets)
        elif self.flags.method == 'iwmopoe':
            model = iwMoPoE(self, self.flags, self.modalities, self.subsets)
        else:
            raise NotImplementedError(f'Method {self.flags.method} not implemented. Exiting...!')
        return model.to(self.flags.device)

    def set_eval_mode(self) -> None:
        """Set all parts of the MMVAE to eval mode."""
        self.mm_vae.eval()

        for mod_k, mod in self.modalities.items():
            mod.encoder.eval()
            mod.decoder.eval()

    def set_train_mode(self) -> None:
        """Set all parts of the MMVAE to eval mode."""
        self.mm_vae.train()

        for mod_k, mod in self.modalities.items():
            mod.encoder.train()
            mod.decoder.train()

    def set_optimizer(self):
        # optimizer definition
        params = []
        for _, mod in self.modalities.items():
            for model in [mod.encoder, mod.decoder]:
                for p in model.parameters():
                    params.append(p)

        # add flow parameters from mmvae if present
        params.extend(list(self.mm_vae.parameters()))

        if hasattr(self.mm_vae, 'flow'):
            for p in ['u', 'w', 'b']:
                if hasattr(self.mm_vae.flow, p):
                    params.append(getattr(self.mm_vae.flow, p))

        optimizer = optim.Adam(params, lr=self.flags.initial_learning_rate, betas=(self.flags.beta_1,
                                                                                   self.flags.beta_2))
        self.flags.num_parameters = len(params)
        self.optimizer = optimizer

    @abstractmethod
    def set_dataset(self):
        pass

    @abstractmethod
    def set_rec_weights(self):
        pass

    @abstractmethod
    def set_style_weights(self):
        pass

    @abstractmethod
    def mean_eval_metric(self, values):
        pass

    @abstractmethod
    def eval_label(self, values, labels, index=None):
        pass

    @abstractmethod
    def set_modalities(self) -> Mapping[str, BaseModality]:
        pass

    def set_subsets(self) -> Mapping[str, Iterable[BaseModality]]:
        """
        powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        >>> exp.modalities = {'a':None, 'b':None, 'c':None}
        >>> exp.set_subsets()
        {'a': [None], 'b': [None], 'c': [None], 'a_b': [None, None], 'a_c': [None, None], 'b_c': [None, None],
        'a_b_c': [None, None, None]}
        """
        xs = list(self.modalities)
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))
        subsets = {}
        for k, mod_names in enumerate(subsets_list):
            mods = [self.modalities[mod_name] for mod_name in sorted(mod_names)]
            key = '_'.join(sorted(mod_names))
            subsets[key] = mods
        return {k: v for k, v in subsets.items() if k != ''}

    def set_paths_fid(self):
        dir_real = os.path.join(self.flags.dir_gen_eval_fid, 'real')
        dir_random = os.path.join(self.flags.dir_gen_eval_fid, 'random')
        paths = {'real': dir_real,
                 'random': dir_random}
        dir_cond = self.flags.dir_gen_eval_fid
        for name in [*self.subsets, 'joint']:
            paths[name] = os.path.join(dir_cond, name)
        return paths

    @staticmethod
    def set_random_seed(deterministic: bool, seed: int):
        if deterministic:
            torch.use_deterministic_algorithms(True)
        # set the seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_test_samples(self, num_images=10) -> typing.Iterable[typing.Mapping[str, Tensor]]:
        """
        Gets random samples for the cond. generation.
        """
        n_test = self.dataset_test.__len__()
        samples = []
        for _ in range(num_images):
            sample, _ = self.dataset_test.__getitem__(random.randint(0, n_test - 1))
            sample = utils.dict_to_device(sample, self.flags.device)

            samples.append(sample)

        return samples
