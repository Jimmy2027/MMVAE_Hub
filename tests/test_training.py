import tempfile
from pathlib import Path

import pytest
import torch
from mmvae_base.modalities.text.alphabet import alphabet
from mmvae_base.utils.plotting import generate_plots

import mmvae_hub
from mmvae_hub import run_epochs
from mmvae_hub.mmnist.experiment import MMNISTExperiment
# from mmvae_hub.mnistsvhntext.flags import flags_set_alpha_modalities, setup_flags, parser
from mmvae_hub.mmnist.flags import FlagsSetup, parser


@pytest.mark.tox
def test_run_epochs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        flags = parser.parse_args([])
        config_path = Path(mmvae_hub.__file__).parent.parent / 'configs/toy_config.json'
        flags_setup = FlagsSetup(config_path)
        flags = flags_setup.setup(flags, testing=True)
        use_cuda = torch.cuda.is_available()
        flags.device = torch.device('cuda' if use_cuda else 'cpu')
        flags.dir_experiment = tmpdirname
        mst = MMNISTExperiment(flags, alphabet)
        mst.set_optimizer()
        run_epochs(mst)


def test_generate_plots():
    with tempfile.TemporaryDirectory() as tmpdirname:
        flags = parser.parse_args([])
        config_path = Path(mmvae_hub.__file__).parent.parent / 'configs/toy_config.json'
        flags_setup = FlagsSetup(config_path)
        flags = flags_setup.setup(flags, testing=True)
        use_cuda = torch.cuda.is_available()
        flags.device = torch.device('cuda' if use_cuda else 'cpu')
        flags.dir_experiment = tmpdirname
        mst = MMNISTExperiment(flags, alphabet)
        generate_plots(mst, epoch=1)


if __name__ == '__main__':
    # pass
    test_run_epochs()
    # test_generate_plots()
