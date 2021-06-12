from mmvae_hub.utils.Dataclasses import BaseTestResults


def get_hyperopt_score(test_results: BaseTestResults, method: str, use_zk: bool) -> int:
    """
    Sum over all metrics to get a score for the optimization of hyperparameters.
    """
    score_gen_eval = sum(score for _, score in test_results.gen_eval.items())
    if use_zk:
        score_lr_eval = sum(score['accuracy'] for _, score in test_results.lr_eval_zk.items())
    else:
        score_lr_eval = sum(score['accuracy'] for _, score in test_results.lr_eval_q0.items())

    return score_lr_eval + score_gen_eval
