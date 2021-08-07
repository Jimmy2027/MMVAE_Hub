import tempfile

import pytest
from norby.utils import get_readable_elapsed_time

from mmvae_hub.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.mimic.MimicTrainer import MimicTrainer
from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.utils.plotting.plotting import generate_plots
from tests.utils import set_me_up


@pytest.mark.tox
@pytest.mark.parametrize("method", ['joint_elbo', 'planar_mixture', 'moe', 'poe', 'pfom', 'pgfm'])
# @pytest.mark.parametrize("method", ['joint_elbo'])
def test_run_epochs_polymnist(method: str):
    """
    Test if the main training loop runs.
    Assert if the total_test loss is constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # todo implement calc likelihood for flow based methods
        calc_nll = False if method in ['planar_mixture', 'pfom', 'pope', 'fomfop', 'fomop', 'poe', 'gfm','planar_vae'] else True
        # calc_nll = False
        mst = set_me_up(tmpdirname, dataset='polymnist', method=method, attributes={'calc_nll': calc_nll,

                                                                                    # 'num_mods': 1
                                                                                    # 'num_flows': 1
                                                                                    })
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()


def test_run_epochs_mimic(method: str):
    """
    Test if the main training loop runs.
    Assert if the total_test loss is constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # todo implement calc likelihood for flow based methods
        calc_nll = method not in ['planar_mixture', 'pfom', 'pope']
        mst = set_me_up(tmpdirname, dataset='mimic', method=method, attributes={'calc_nll': True,
                                                                                'use_clf': True,
                                                                                # 'calc_prd': True
                                                                                })
        trainer = MimicTrainer(mst)
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
    from time import time

    start_time = time()
    # test_run_epochs_polymnist(method='afom')
    # test_run_epochs_polymnist(method='pfom')
    # test_run_epochs_polymnist(method='gfm')
    # test_run_epochs_polymnist(method='gfmop')
    # test_run_epochs_polymnist(method='fomop')
    test_run_epochs_polymnist(method='planar_vae')
    # test_run_epochs_polymnist(method='mogfm')
    # test_run_epochs_mimic(method='joint_elbo')
    # test_run_epochs_polymnist(method='iwmogfm')

    # test_run_epochs_polymnist(method='mogfm')
    elapsed_time = time() - start_time
    print(get_readable_elapsed_time(elapsed_time))

    # test_run_epochs_polymnist(method='mofogfm')

    # test_run_epochs_polymnist(method='pope')
    # test_run_planar_mixture_no_flow()
    # test_generate_plots()
    # test_test_generation()
