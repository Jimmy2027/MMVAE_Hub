# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mmvae_hub.base.utils.utils import json2dict, write_to_jsonfile
from mmvae_hub.polymnist import PolymnistTrainer
from tests.utils import set_me_up


@pytest.mark.tox
@pytest.mark.parametrize("method", ['moe', 'joint_elbo'])
def test_static_results_1mod(method: str, update_static_results=False):
    """
    Test if the results are constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    jsonfile = Path(__file__).parent / 'static_results.json'
    static_results = json2dict(jsonfile)['static_results_1mod'][method]

    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname,
                        method=method,
                        attributes={'num_flows': 0, 'num_mods': 1, 'deterministic': True, 'device': 'cpu',
                                    'steps_per_training_epoch': 1, 'factorized_representation': False})
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()

        if update_static_results:
            static_results['joint_div'] = test_results.joint_div
            static_results['klds'] = test_results.klds['m0']
            static_results['lhoods'] = test_results.lhoods['m0']['m0']
            static_results['log_probs'] = test_results.log_probs['m0']
            static_results['total_loss'] = test_results.total_loss
            static_results['lr_eval'] = test_results.lr_eval['m0']['accuracy']
            static_results['latents_class'] = {}
            static_results['latents_class']['mu'] = test_results.latents['m0']['latents_class']['mu']

            write_to_jsonfile(jsonfile, [(f'static_results_1mod.{method}', static_results)])

        are_they_equal = {
            'joint_div': np.round(test_results.joint_div, 5) == np.round(static_results['joint_div'], 5),
            'klds': np.round(test_results.klds['m0'], 5) == np.round(static_results['klds'], 5),
            'lhoods': np.round(test_results.lhoods['m0']['m0'], 5) == np.round(static_results['lhoods'], 5),
            'log_probs': test_results.log_probs['m0'] == static_results['log_probs'],
            'total_loss': test_results.total_loss == static_results['total_loss'],
            'lr_eval': test_results.lr_eval['m0']['accuracy'] == static_results['lr_eval'],
            'latents_class_mu': np.round(test_results.latents['m0']['latents_class']['mu'], 8) == np.round(
                static_results['latents_class']['mu'], 8)
        }

        assert all(v for _, v in are_they_equal.items()), f'Some results changed: {are_they_equal}'


# @pytest.mark.tox
@pytest.mark.parametrize("method", ['moe', 'joint_elbo'])
def test_static_results_2mods(method: str):
    """
    Test if the results are constant. If the assertion fails, it means that the model or the evaluation has
    changed, perhaps involuntarily.
    """
    static_results = json2dict(Path('static_results.json'))['static_results_2mod']

    with tempfile.TemporaryDirectory() as tmpdirname:
        mst = set_me_up(tmpdirname,
                        method=method,
                        attributes={'num_flows': 0, 'num_mods': 2, 'deterministic': True, 'device': 'cpu',
                                    'steps_per_training_epoch': 1, 'factorized_representation': False})
        trainer = PolymnistTrainer(mst)
        test_results = trainer.run_epochs()
        assert np.round(test_results.joint_div, 1) == np.round(static_results[method]['joint_div'], 1)
        assert np.round(test_results.klds['m0'], 1) == np.round(static_results[method]['klds'], 1)
        assert np.round(test_results.lhoods['m0']['m0'], 1) == np.round(static_results[method]['lhoods'], 1)
        assert np.round(test_results.log_probs['m0'], 0) == np.round(static_results[method]['log_probs'], 0)
        assert np.round(test_results.total_loss, 0) == np.round(static_results[method]['total_loss'], 0)
        assert np.round(test_results.lr_eval['m0']['accuracy'], 2) == np.round(static_results[method]['lr_eval'], 2)
        assert np.round(test_results.latents['m0']['latents_class']['mu'], 2) == np.round(
            static_results[method]['latents_class']['mu'], 2)


if __name__ == '__main__':
    test_static_results_1mod('moe', update_static_results=False)
    test_static_results_1mod('joint_elbo', update_static_results=False)
