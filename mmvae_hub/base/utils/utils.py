import collections
import itertools
import json
import os
import subprocess as sp
from collections.abc import MutableMapping
from pathlib import Path
from typing import Iterable, Optional, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch import device as Device
from torch.autograd import Variable

from mmvae_hub import log


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


def reweight_weights(w):
    return w / w.sum()


def get_items_from_dict(in_dict: Mapping[str, Tensor]) -> Mapping[str, float]:
    return {k1: v1.cpu().item() for k1, v1 in in_dict.items()}


def mixture_component_selection(flags, mus, logvars, w_modalities=None, num_samples=None):
    num_components = mus.shape[0]
    # num_samples is the batch_size
    num_samples = mus.shape[1]
    # if not defined, take pre-defined weights
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device)
    idx_start = []
    idx_end = []
    for k in range(num_components):
        i_start = 0 if k == 0 else int(idx_end[k - 1])
        if k == w_modalities.shape[0] - 1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)

    idx_end[-1] = num_samples

    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    return [mu_sel, logvar_sel]


def flow_mixture_component_selection(flags, reps, w_modalities=None, num_samples=None):
    # if not defined, take pre-defined weights
    num_samples = reps.shape[1]
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device)
    idx_start = []
    idx_end = []
    for k in range(w_modalities.shape[0]):
        i_start = 0 if k == 0 else int(idx_end[k - 1])
        if k == w_modalities.shape[0] - 1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)

    idx_end[-1] = num_samples
    return torch.cat(
        [
            reps[k, idx_start[k]: idx_end[k], :]
            for k in range(w_modalities.shape[0])
        ]
    )


def save_and_log_flags(flags) -> str:
    filename_flags_rar = os.path.join(flags.dir_experiment_run, 'flags.rar')
    torch.save(flags, filename_flags_rar)
    str_args = ''
    for k, key in enumerate(sorted(flags.__dict__.keys())):
        str_args = str_args + '\n' + key + ': ' + str(flags.__dict__[key])
    return str_args


def get_clf_path(clf_dir: str, clf_name: str) -> Optional[str]:
    """
    Since the total training epochs of the classifier is not known but is in its filename, the filename needs to be
    found by scanning the directory.
    """
    for file in os.listdir(clf_dir):
        filename = '_'.join(file.split('_')[:-1])
        if filename == clf_name:
            return os.path.join(clf_dir, file)
    else:
        raise FileNotFoundError(f'No {clf_name} classifier was found in {clf_dir}')


def get_alphabet(alphabet_path=Path(__file__).parent.parent / 'alphabet.json'):
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    return alphabet


def at_most_n(X, n):
    """
    Yields at most n elements from iterable X. If n is None, iterates until the end of iterator.
    """
    yield from itertools.islice(iter(X), n)


def set_up_process_group(world_size: int, rank) -> None:
    """
    sets up a process group for distributed training.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)


def get_gpu_memory():
    """
    Taken from:
    https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    """

    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    return [int(x.split()[0]) for i, x in enumerate(memory_free_info)]


class NaNInLatent(Exception):
    pass


def check_latents(dataset: str, latents):
    """
    checks if the latents contain NaNs. If they do raise NaNInLatent error and the experiment is started again
    """
    if dataset != 'testing' and (np.isnan(latents[0].mean().item())
                                 or
                                 np.isnan(latents[1].mean().item())):
        raise NaNInLatent(f'The latent representations contain NaNs: {latents}')


def flatten(d: dict, parent_key='', sep='_') -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def write_to_jsonfile(config_path: Path, parameters: list):
    """
    parameters: list of tuples. Example [('model.use_cuda',VALUE),] where VALUE is the parameter to be set
    """
    with open(config_path) as file:
        config = json.load(file)
    for parameter, value in parameters:
        split = parameter.split('.')
        key = config[split[0]]
        for idx in range(1, len(split) - 1):
            key = key[split[idx]]
        key[split[-1]] = value

    with open(config_path, 'w') as outfile:
        json.dump(config, outfile, indent=4)


def stdout_if_verbose(verbose: int, message, min_level: int):
    """
    verbose: current global verbose setting
    message: message to be sent to stdout
    level: minimum verbose level needed to send the message
    """
    if verbose >= min_level:
        log.info(message)


def dict_to_device(d: dict, dev: Device):
    return {k: v.to(dev) for k, v in d.items()}


def init_twolevel_nested_dict(level1_keys: list, level2_keys: list, init_val: any, copy_init_val: bool = False) -> dict:
    """
    Initialise a 2 level nested dict with value: init_val.
    copy_init_val: when using a list need to copy value.
    """
    if copy_init_val:
        return {l1: {l2: init_val.copy() for l2 in level2_keys if l2} for l1 in level1_keys if l1}
    else:
        return {l1: {l2: init_val for l2 in level2_keys if l2} for l1 in level1_keys if l1}


def get_items_from_nested_dict(nested: dict) -> dict:
    new = {}
    for k1, v1 in nested.items():
        new[k1] = {}
        for k2, v2 in v1.items():
            new[k1] = v2.cpu().item()

    return new


class OnlyOnce:
    """
    Contains a set of strings. If the passed string is not in the set, returns True.
    """

    def __init__(self):
        self.myset: Iterable[str] = {''}

    def __call__(self, arg: str):
        if arg in self.myset:
            return False
        self.myset.add(arg)
        return True


def imshow(img):
    import matplotlib.pyplot as plt
    img = img.squeeze()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def json_file_to_pyobj(filename: str):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())

    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def atleast_2d(tensor, dim: int):
    if len(tensor.shape) < 2:
        return tensor.unsqueeze(dim)


def dict2json(out_path: Path, d: dict):
    with open(out_path, 'w') as outfile:
        json.dump(d, outfile, indent=2)


def json2dict(json_path: Path) -> dict:
    with open(json_path, 'rt') as json_file:
        json_config = json.load(json_file)
    return json_config
