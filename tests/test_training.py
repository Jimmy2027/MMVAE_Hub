import tempfile
from pathlib import Path

import pytest

import mmvae_hub
from mmvae_hub.base.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.mmnist import MmnistTrainer
from mmvae_hub.mmnist.experiment import MMNISTExperiment
from mmvae_hub.mmnist.flags import FlagsSetup, parser


def set_me_up(tmpdirname):
    flags = parser.parse_args([])
    config_path = Path(mmvae_hub.__file__).parent.parent / 'configs/toy_config.json'
    flags_setup = FlagsSetup(config_path)
    flags = flags_setup.setup_test(flags, tmpdirname)
    mst = MMNISTExperiment(flags)
    mst.set_optimizer()
    return mst


@pytest.mark.tox
def test_run_epochs_mmnist():
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname)
        trainer = MmnistTrainer(mst)
        trainer.run_epochs()


def test_generate_plots():
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname)
        generate_plots(mst, epoch=1)


def test_test_generation():
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname)
        test_generation(mst)


if __name__ == '__main__':
    # pass
    test_run_epochs_mmnist()
    # test_generate_plots()
    # test_test_generation()
