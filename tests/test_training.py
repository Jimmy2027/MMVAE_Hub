import tempfile
from pathlib import Path

import pytest
from norby.utils import get_readable_elapsed_time

from mmvae_hub.evaluation.eval_metrics.coherence import test_generation
from mmvae_hub.mimic.MimicTrainer import MimicTrainer
from mmvae_hub.mnistsvhntext.mnistsvhntextTrainer import mnistsvhnTrainer
from mmvae_hub.polymnist.PolymnistTrainer import PolymnistTrainer
from mmvae_hub.utils.plotting.plotting import generate_plots
from tests.utils import set_me_up


@pytest.mark.tox
@pytest.mark.parametrize("method", ['mopoe', 'moe', 'poe', 'mopgfm', 'iwmogfm'])
# @pytest.mark.parametrize("method", ['joint_elbo'])
def test_run_epochs_polymnist(method: str):
    """
    Test if the main training loop runs.
    Assert if the total_test loss is constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # todo implement calc likelihood for flow based methods
        calc_nll = False if method in ['planar_mixture', 'pfom', 'pope', 'fomfop', 'fomop', 'poe', 'gfm', 'planar_vae',
                                       'sylvester_vae_noflow', 'iwmogfm', 'iwmogfm2', 'iwmogfm3','iwmogfm_amortized','iwmogfm_old'] else True
        # calc_nll = False
        mst = set_me_up(tmpdirname, dataset='polymnist', method=method, attributes={'calc_nll': calc_nll,
                                                                                    "K": 5,
                                                                                    "dir_clf": Path("/tmp/trained_clfs_polyMNIST")

                                                                                    # 'num_mods': 1
                                                                                    # 'num_flows': 1
                                                                                    })
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()


# @pytest.mark.tox
# @pytest.mark.parametrize("method", ['mopoe'])
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
                                                                                'batch_size':2,
                                                                                # 'calc_prd': True
                                                                                })
        trainer = MimicTrainer(mst)
        test_results = trainer.run_epochs()


def test_run_epochs_mnistsvhntext(method: str):
    """
    Test if the main training loop runs.
    Assert if the total_test loss is constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # todo implement calc likelihood for flow based methods
        calc_nll = method not in ['planar_mixture', 'pfom', 'pope']
        mst = set_me_up(tmpdirname, dataset='mnistsvhntext', method=method, attributes={'calc_nll': True,
                                                                                        "dir_clf": Path("/tmp/trained_clfs_mst")
                                                                                        })
        trainer = mnistsvhnTrainer(mst)
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
    # test_run_epochs_mimic(method='mopoe')
    test_run_epochs_polymnist(method='iwmogfm_amortized')
    # test_run_epochs_polymnist(method='iwmogfm2')
    # test_run_epochs_polymnist(method='iwmogfm')
    # test_run_epochs_mnistsvhntext(method='mopoe')
    # test_run_epochs_polymnist(method='iwmogfm_amortized')
    # test_run_epochs_polymnist(method='iwmogfm2')
    # test_run_epochs_mimic(method='iwmogfm')
    # test_run_epochs_mnistsvhntext(method='mopoe')
    # test_run_epochs_polymnist(method='iwmopgfm')
    # test_run_epochs_polymnist(method='mopgfm')
    # test_run_epochs_polymnist(method='mopoe')
    # test_run_epochs_polymnist(method='iwmopoe')
    # test_run_epochs_polymnist(method='iwmoe')

    # test_run_epochs_polymnist(method='mogfm')
    elapsed_time = time() - start_time
    print(get_readable_elapsed_time(elapsed_time))

    # test_run_epochs_polymnist(method='mofogfm')

    # test_run_epochs_polymnist(method='pope')
    # test_run_planar_mixture_no_flow()
    # test_generate_plots()
    # test_test_generation()
