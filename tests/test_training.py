import tempfile
from pathlib import Path

import pytest

import mmvae_hub
from mmvae_hub.base.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.polymnist import PolymnistExperiment
from mmvae_hub.polymnist import PolymnistTrainer
from mmvae_hub.polymnist.flags import FlagsSetup, parser


def set_me_up(tmpdirname):
    flags = parser.parse_args([])
    config_path = Path(mmvae_hub.__file__).parent.parent / 'configs/toy_config.json'
    flags_setup = FlagsSetup(config_path)
    flags = flags_setup.setup_test(flags, tmpdirname)
    flags.method = 'joint_elbo'
    # flags.method = 'planar_mixture'
    mst = PolymnistExperiment(flags)
    mst.set_optimizer()
    return mst


@pytest.mark.tox
def test_run_epochs_polymnist():
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname)
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()
        assert test_results['total_loss'] == 7733.9169921875


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
    test_run_epochs_polymnist()
    # test_generate_plots()
    # test_test_generation()
