# -*- coding: utf-8 -*-
import argparse
import configparser
import os
import tempfile
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch

import mmvae_hub
from mmvae_hub import log
from mmvae_hub.utils.MongoDB import MongoDatabase
from mmvae_hub.utils.setup.filehandling import create_dir_structure, get_experiment_uid
from mmvae_hub.utils.utils import json2dict, unpack_zipfile, dict2pyobject


class BaseFlagsSetup:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.parser = None

    def setup(self, flags, testing=False, additional_args: dict = None):
        """
        leomed bool: if True, use TMPDIR as experiment_dir and dir_data
        Setup the flags:
            - update_flags_with_config
            - expand user in paths and set paths if not given
            - set device
            - set alpha modalities
            - set len_sequence
            - load flags
            - set seed
        """
        flags.config_path = self.config_path
        if self.config_path:
            flags = update_flags_with_config(p=self.parser, config_path=flags.config_path, testing=testing)

        if additional_args:
            for k, v in additional_args.items():
                setattr(flags, k, v)

        if flags.calc_prd:
            # calc_prd needs saved figures
            flags.save_figure = True

        if not flags.experiment_uid:
            experiment_uid = get_experiment_uid(flags.exp_str_prefix, flags.method)
            flags.experiment_uid = experiment_uid

        if not flags.dir_fid:
            flags.dir_fid = flags.dir_experiment

        if flags.leomed:
            flags = self.setup_leomed(flags)

        flags.version = self.get_version_from_setup_config()

        flags = self.setup_paths(flags)

        flags = create_dir_structure(flags)

        use_cuda = flags.use_cuda and not flags.deterministic and torch.cuda.is_available()
        flags.device = torch.device('cuda' if use_cuda else 'cpu')

        if use_cuda:
            torch.cuda.set_device(get_freer_gpu())

        flags = self.flags_set_alpha_modalities(flags)

        flags.log_file = Path(log.manager.root.handlers[1].baseFilename)

        if flags.load_flags:
            old_flags = torch.load(Path(flags.load_flags).expanduser())
            # create param dict from all the params of old_flags that are not paths
            params = {k: v for k, v in old_flags.item() if ('dir' not in v) and ('path' not in v)}
            flags.__dict__.update(params)

        if not flags.seed:
            # set a random seed
            flags.seed = np.random.randint(0, 10000)

        return flags

    @abstractmethod
    def flags_set_alpha_modalities(self, flags):
        pass

    @staticmethod
    def setup_paths(flags: argparse.ArgumentParser()) -> argparse.ArgumentParser():
        """Expand user in paths and set dir_fid if not given."""
        flags.dir_data = Path(flags.dir_data).expanduser()
        flags.dir_experiment = Path(flags.dir_experiment).expanduser()
        flags.inception_state_dict = Path(flags.inception_state_dict).expanduser()
        flags.dir_fid = Path(flags.dir_fid).expanduser() if flags.dir_fid else flags.dir_experiment / 'fid'
        flags.dir_clf = Path(flags.dir_clf).expanduser() if flags.use_clf else None

        assert flags.dir_data.exists() or flags.dataset == 'toy', f'data path: "{flags.dir_data}" not found.'
        return flags

    def setup_test(self, flags, tmpdirname: str):
        flags = self.setup(flags, testing=True)
        flags.dir_experiment = Path(tmpdirname) / 'tmpexp'
        flags.dir_fid = Path(tmpdirname) / 'fid'
        return flags

    @staticmethod
    def get_version_from_setup_config() -> str:
        """Read the package version from the setup.cfg file."""
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.parent.parent.parent / 'setup.cfg')
        return config['metadata']['version']

    def setup_leomed(self, flags):
        tmpdir = Path(os.getenv("TMPDIR"))

        # unzip polymnist dataset to tmpdir
        if flags.dataset == 'polymnist':
            polymnist_zip_path = Path('/cluster/work/vogtlab/Projects/Polymnist/PolyMNIST.zip')
            out_dir = tmpdir

            log.info(f'Extracting data from {polymnist_zip_path} to {out_dir}.')
            unpack_zipfile(polymnist_zip_path, out_dir)

            flags.dir_data = out_dir / 'PolyMNIST'

            assert out_dir.exists(), f'Data dir {out_dir} does not exist.'

        flags.dir_fid = tmpdir / 'fid'

        flags.dir_experiment = tmpdir

        return flags

    @staticmethod
    def set_paths_with_config(config: dict, flags, is_dict: bool = False):
        """
        Update paths in flags with paths from config.
        Use this if experiment was run on another machine to update the paths in flags.
        Since the attributes of the flag object cannot be set, the input "flags" needs to be a dict.
        """

        for key in config:
            if key in ['dir_experiment', 'dir_clf', 'dir_data']:
                if is_dict:
                    flags[key] = Path(config[key]).expanduser()
                else:
                    setattr(flags, key, Path(config[key]).expanduser())

        return flags

    def load_old_flags(self, flags_path: Path = None, _id: str = None, is_dict: bool = False, add_args: dict = None):
        """
        Load flags from old experiments, either from a directory or from the db.
        Add parameters for backwards compatibility and adapt paths for current system.

        If flags_path is None, flags will be loaded from the db using the _id.
        """
        defaults = [('weighted_mixture', False), ('amortized_flow', False), ('coupling_dim', 512), ('beta_warmup', 0),
                    ('vocab_size', 2900), ('nbr_coupling_block_layers', 0)]
        add_args = add_args | {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}

        if is_dict or flags_path is None:
            if flags_path is None:
                # get flags from db
                db = MongoDatabase(_id=_id)
                flags = db.get_experiment_dict()['flags']
            else:
                # load flags from jsonfile
                flags = json2dict(flags_path)
            flags = self.set_paths_with_config(json2dict(self.config_path), flags, True)

            # get defaults from newer parameters that might not be defined in old flags
            for k, v in defaults:
                if k not in flags:
                    flags[k] = v

            if add_args is not None:
                for k, v in add_args.items():
                    flags[k] = v

            if 'min_beta' not in flags:
                flags['min_beta'] = flags['beta']
                flags['max_beta'] = flags['beta']

            if 'num_gfm_flows' not in flags:
                flags['num_gfm_flows'] = flags['num_flows']

            # becomes immutable..
            flags = dict2pyobject(flags, 'flags')

        else:
            # load flags from .rar file
            flags = torch.load(flags_path)
            flags = self.set_paths_with_config(json2dict(self.config_path), flags, False)

            # get defaults from newer parameters that might not be defined in old flags
            for k, v in defaults:
                if not hasattr(flags, k):
                    setattr(flags, k, v)

            if add_args is not None:
                for k, v in add_args.items():
                    setattr(flags, k, v)

            if not hasattr(flags, 'min_beta'):
                setattr(flags, 'min_beta', flags.beta)
                setattr(flags, 'max_beta', flags.beta)

            if not hasattr(flags, 'num_gfm_flows'):
                setattr(flags, 'num_gfm_flows', flags.num_flows)

        return flags


def get_freer_gpu() -> int:
    """
    Returns the index of the gpu with the most free memory.
    Taken from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.system(f'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{Path(tmpdirname) / "tmp"}')
        memory_available = [int(x.split()[2]) for x in open(Path(tmpdirname) / "tmp", 'r').readlines()]
    return int(np.argmax(memory_available))


def update_flags_with_config(p, config_path: Path, additional_args: dict = None, testing=False):
    """
    If testing is true, no cli arguments will be read.

    Parameters
    ----------
    p : parser to be updated.
    config_path : path to the json config file.
    additional_args : optional additional arguments to be passed as dict.
    """
    additional_args = additional_args or {}
    json_config = json2dict(config_path)
    t_args = argparse.Namespace()
    t_args.__dict__.update({**json_config, **additional_args})
    if testing:
        return p.parse_args([], namespace=t_args)
    else:
        return p.parse_args(namespace=t_args)


def get_config_path(dataset: str = None, flags=None):
    dataset = dataset or flags.dataset

    if flags and flags.config_path:
        return flags.config_path

    assert dataset, f'Dataset parameter must be given either through function input or through flags.'

    if os.path.exists('/cluster/home/klugh/'):
        return os.path.join(Path(os.path.dirname(mmvae_hub.__file__)).parent,
                            f"configs/{dataset}/leomed_config.json")

    elif os.path.exists('/mnt/data/hendrik'):
        return os.path.join(Path(os.path.dirname(mmvae_hub.__file__)).parent,
                            f"configs/{dataset}/bartholin_config.json")

    else:
        return os.path.join(Path(os.path.dirname(mmvae_hub.__file__)).parent,
                            f"configs/{dataset}/local_config.json")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
