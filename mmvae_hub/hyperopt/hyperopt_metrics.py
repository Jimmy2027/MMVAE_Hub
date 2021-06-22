import numpy as np

from mmvae_hub import log
from mmvae_hub.utils.Dataclasses import BaseTestResults


def get_hyperopt_score(test_results: BaseTestResults, method: str, use_zk: bool) -> int:
    """
    Sum over all metrics to get a score for the optimization of hyperparameters.
    """
    score_gen_eval = np.mean([score for _, score in test_results.gen_eval.items()])
    score_fid = np.mean([score for _, score in test_results.prd_scores.items()])
    if use_zk:
        score_lr_eval = np.mean([score['accuracy'] for _, score in test_results.lr_eval_zk.items()])
    else:
        score_lr_eval = np.mean([score['accuracy'] for _, score in test_results.lr_eval_q0.items()])

    score = score_lr_eval / 0.75 + score_gen_eval / 0.55 + score_fid / 0.1
    log.info(f'Current hyperopt score is {score}. score_lr_eval: {score_lr_eval}, score_gen_eval: {score_gen_eval}, '
             f'score_fid: {score_fid}')
    return score
