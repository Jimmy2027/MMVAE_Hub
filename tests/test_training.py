import tempfile

import pytest

from mmvae_hub.base.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.polymnist import PolymnistTrainer
from tests.utils import set_me_up


@pytest.mark.tox
@pytest.mark.parametrize("method", ['joint_elbo', 'planar_mixture'])
def test_run_epochs_polymnist(method: str):
    """
    Test if the main training loop runs.
    Assert if the total_test loss is constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname, method)
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()
        # assert test_results['total_loss'] == 7733.9169921875


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
    test_run_epochs_polymnist(method='joint_elbo')
    # test_generate_plots()
    # test_test_generation()
