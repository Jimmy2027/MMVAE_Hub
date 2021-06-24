import numpy as np

from mmvae_hub import log
from mmvae_hub.utils.Dataclasses import BaseTestResults


def get_hyperopt_score(test_results: BaseTestResults, method: str, use_zk: bool, optuna_trial):
    """
    Sum over all metrics to get a score for the optimization of hyperparameters.
    """
    score_gen_eval = np.mean([score for _, score in test_results.gen_eval.items()])
    score_prd = np.mean([score for _, score in test_results.prd_scores.items()])

    if use_zk:
        score_lr_eval = np.mean([score['accuracy'] for _, score in test_results.lr_eval_zk.items()])
    else:
        score_lr_eval = np.mean([score['accuracy'] for _, score in test_results.lr_eval_q0.items()])

    optuna_trial.suggest_float("score_gen_eval", score_gen_eval, score_gen_eval)
    optuna_trial.suggest_float("score_prd", score_prd, score_prd)
    optuna_trial.suggest_float("score_lr_eval", score_lr_eval, score_lr_eval)

    score = score_lr_eval / 0.9 + score_gen_eval / 0.55 + score_prd / 0.05
    # score = score_lr_eval / 0.75 + score_gen_eval / 0.55 + score_prd / 0.1
    log.info(f'Current hyperopt score is {score}. score_lr_eval: {score_lr_eval}, score_gen_eval: {score_gen_eval}, '
             f'score_prd: {score_prd}')
    return optuna_trial, score
