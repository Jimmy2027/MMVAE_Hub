# -*- coding: utf-8 -*-
import argparse
import configparser
import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch

import mmvae_hub
from mmvae_hub import log
from mmvae_hub.base.utils.filehandling import create_dir_structure, get_experiment_uid
from mmvae_hub.base.utils.utils import json2dict, unpack_zipfile
from mmvae_hub.polymnist.experiment import PolymnistExperiment


class BaseFlagsSetup:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.parser = None

    def setup(self, flags, testing=False, additional_args=None):
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

        experiment_uid = get_experiment_uid(flags)
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
        polymnist_zip_path = Path('/cluster/work/vogtlab/Projects/Polymnist/PolyMNIST.zip')
        tmpdir = Path(os.getenv("TMPDIR"))
        out_dir = tmpdir

        log.info(f'Extracting data from {polymnist_zip_path} to {out_dir}.')
        unpack_zipfile(polymnist_zip_path, out_dir)

        flags.dir_data = out_dir / 'PolyMNIST'

        assert out_dir.exists(), f'Data dir {out_dir} does not exist.'

        flags.dir_fid = tmpdir / 'fid'

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


def get_freer_gpu() -> int:
    """
    Returns the index of the gpu with the most free memory.
    Taken from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/6
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
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


def get_config_path(flags=None):
    if not flags or not flags.config_path:
        if os.path.exists('/cluster/home/klugh/'):
            return os.path.join(Path(os.path.dirname(mmvae_hub.__file__)).parent, "configs/leomed_config.json")
        elif os.path.exists('/mnt/data/hendrik'):
            return os.path.join(Path(os.path.dirname(mmvae_hub.__file__)).parent, "configs/bartholin_config.json")
        else:
            return os.path.join(Path(os.path.dirname(mmvae_hub.__file__)).parent, "configs/local_config.json")
    else:
        return flags.config_path


def get_experiment(flags):
    if Path(flags.dir_data).name in ['PolyMNIST', 'polymnist']:
        return PolymnistExperiment(flags)
    elif flags.dataset == 'toy':
        return PolymnistExperiment(flags)
    else:
        raise RuntimeError(f'No experiment for {Path(flags.dir_data).name} implemented.')
