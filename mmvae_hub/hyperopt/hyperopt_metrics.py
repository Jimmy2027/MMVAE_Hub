import numpy as np

from mmvae_hub import log
from mmvae_hub.utils.Dataclasses.Dataclasses import BaseTestResults


def get_missing_mod_scores_gen_eval(results: dict):
    """Get the scores that were achieved with a missing modality."""
    for key, score in results.items():
        if 'joint' not in key:
            # separate key between str_label, in_mods and out_mods
            split1 = key.split('__')
            out_mod = split1[-1]
            split2 = split1[0].split('_')
            in_mods = split2[1:]

            if key == 'random':
                yield np.mean([v for _, v in score.items()])

            elif out_mod not in in_mods:
                yield score


def get_reconstr_mod_scores_gen_eval(results: dict):
    """Get the scores that were achieved with the modality given as input."""
    for key, score in results.items():
        if 'joint' not in key:
            # separate key between str_label, in_mods and out_mods
            split1 = key.split('__')
            out_mod = split1[-1]
            split2 = split1[0].split('_')
            in_mods = split2[1:]

            if out_mod in in_mods:
                yield score


def get_missing_mod_scores_prd(results: dict):
    """Get the scores that were achieved with a missing modality."""
    for key, score in results.items():
        if 'joint' not in key:
            # separate key between in_mods and out_mods
            split = key.split('_')
            out_mod = split[-1]
            in_mods = split[:-1]

            if key == 'random':
                yield np.mean([v for _, v in score.items()])

            elif out_mod not in in_mods:
                yield score


def get_reconstr_mod_scores_prd(results: dict):
    """Get the scores that were achieved with the modality given as input."""
    for key, score in results.items():
        # separate key between in_mods and out_mods
        split = key.split('_')
        out_mod = split[-1]
        in_mods = split[:-1]

        if out_mod in in_mods:
            yield score


def get_hyperopt_score(test_results: BaseTestResults, use_zk: bool, optuna_trial):
    """
    Sum over all metrics to get a score for the optimization of hyperparameters.
    """

    # score_gen_eval = np.mean([score for _, score in test_results.gen_eval.items()])
    score_gen_eval = np.mean([score for score in get_missing_mod_scores_gen_eval(test_results.gen_eval)])
    # score_prd = np.mean([score for _, score in test_results.prd_scores.items()])
    score_prd = np.mean([score for score in get_missing_mod_scores_prd(test_results.prd_scores)])

    # if use_zk:
    #     score_lr_eval = np.mean([score['accuracy'] for _, score in test_results.lr_eval_zk.items()])
    # else:
    #     score_lr_eval = np.mean([score['accuracy'] for _, score in test_results.lr_eval_q0.items()])

    # add metrics to optuna so that they can be retrieved in the database.
    optuna_trial.suggest_float("score_gen_eval", score_gen_eval, score_gen_eval)
    optuna_trial.suggest_float("score_prd", score_prd, score_prd)
    # optuna_trial.suggest_float("score_lr_eval", score_lr_eval, score_lr_eval)

    # score = score_lr_eval / 0.9 + score_gen_eval / 0.55 + score_prd / 0.05
    score = score_gen_eval / 0.55 + score_prd / 0.05
    # score = score_lr_eval / 0.75 + score_gen_eval / 0.55 + score_prd / 0.1
    log.info(f'Current hyperopt score is {score}. '
             # f'score_lr_eval: {score_lr_eval}, '
             f'score_gen_eval: {score_gen_eval}, '
             f'score_prd: {score_prd}')
    return optuna_trial, score
