import tempfile

import pytest

from mmvae_hub.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.base.utils.plotting import generate_plots
from mmvae_hub.polymnist import PolymnistTrainer
from tests.utils import set_me_up


@pytest.mark.tox
@pytest.mark.parametrize("method", ['joint_elbo', 'planar_mixture'])
# @pytest.mark.parametrize("method", ['joint_elbo'])
def test_run_epochs_polymnist(method: str):
    """
    Test if the main training loop runs.
    Assert if the total_test loss is constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:

        # todo implement calc likelihood for flow based methods
        calc_nll = False if method in ['planar_mixture', 'pfom'] else True
        mst = set_me_up(tmpdirname, method, attributes={'calc_nll': calc_nll})
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()


# @pytest.mark.tox
def test_run_planar_mixture_no_flow():
    """
    Test if the main training loop runs.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        method = 'planar_mixture'
        additional_attrs = {'num_flows': 0, 'num_mods': 1}
        mst = set_me_up(tmpdirname, method, attributes=additional_attrs)
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()


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

    # test_run_epochs_polymnist(method='joint_elbo')
    # test_run_epochs_polymnist(method='moe')
    # test_run_epochs_polymnist(method='planar_mixture')
    test_run_epochs_polymnist(method='pfom')
    # test_run_planar_mixture_no_flow()
    # test_generate_plots()
    # test_test_generation()
