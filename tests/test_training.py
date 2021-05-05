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
        asfd = 0
        # assert test_results['total_loss'] == 7733.9169921875


@pytest.mark.tox
def test_planar_mixture_no_flow():
    """
    Test if MOE method results in the same results than planar_mixture method with no flows.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # mst = set_me_up(tmpdirname, method='moe',
        #                 attributes={'num_flows': 0, 'num_mods': 1, 'deterministic': True, 'device': 'cpu'})
        mst = set_me_up(tmpdirname,
                        # method='planar_mixture',
                        method='moe',
                        attributes={'num_flows': 0, 'num_mods': 1, 'deterministic': True, 'device': 'cpu',
                                    'steps_per_training_epoch': 1})
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()
        assert test_results.joint_div == 0.2534313499927521
        assert test_results.latents['m0']['latents_class']['mu'] == -0.0005813924944959581
        agfadfgds = 0


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
    test_planar_mixture_no_flow()
    # test_run_epochs_polymnist(method='joint_elbo')
    # test_generate_plots()
    # test_test_generation()
