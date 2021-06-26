from typing import List, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmvae_hub import log
from mmvae_hub.networks.FlowVaes import FlowOfEncModsVAE, FlowVAE, FlowOfJointVAE
from mmvae_hub.utils.Dataclasses import *
from mmvae_hub.utils.utils import dict_to_device, atleast_2d, init_twolevel_nested_dict


def train_clf_lr_all_subsets(exp):
    """
    Encodes samples from the training set and train line classifiers from them.
    """
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    subsets = exp.subsets

    train_loader = DataLoader(exp.dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.dataloader_workers)

    training_steps = exp.flags.steps_per_training_epoch

    all_labels, data_train = get_lr_training_data(args, exp, mm_vae, subsets, train_loader, training_steps)

    # get random labels such that it contains both classes
    # labels, rand_ind_train = get_random_labels(all_labels.shape[0], n_train_samples, all_labels)
    n_train_samples = exp.flags.num_training_samples_lr
    rand_ind_train = np.random.randint(all_labels.shape[0], size=n_train_samples)
    labels = all_labels[rand_ind_train]

    for l_key in ['q0', 'zk']:
        for s_key in [*subsets, 'joint']:
            d = data_train[l_key][s_key]
            data_train[l_key][s_key] = d[rand_ind_train] if len(d) else None

    lr_results_q0 = train_clf_lr(exp, data_train['q0'], labels) if not isinstance(mm_vae, FlowOfEncModsVAE) else None
    lr_results_zk = train_clf_lr(exp, data_train['zk'], labels) \
        if isinstance(mm_vae, FlowVAE) else None

    return lr_results_q0, lr_results_zk


def get_lr_training_data(args, exp, mm_vae, subsets: List[str], train_loader, training_steps):
    """
    Get the latent embedding as the training data for the linear classifiers.
    """
    data_train = init_twolevel_nested_dict(level1_keys=['q0', 'zk'], level2_keys=[*subsets, 'joint'],
                                           init_val=torch.Tensor())
    all_labels = torch.Tensor()
    log.info(f"Creating {training_steps} batches of the latent representations for the classifier.")
    for it, (batch_d, batch_l) in tqdm(enumerate(train_loader), total=len(train_loader),
                                       postfix='creating_train_lr'):
        """
        Constructs the training set (labels and inferred subsets) for the classifier training.
        """
        batch_d = {k: v.to(exp.flags.device) for k, v in batch_d.items()}
        _, joint_latent = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        joint_latent: Union[JointLatents, JointLatentsFoEM, JointLatentsFoJ, JointLatentsFoS]

        all_labels = torch.cat((all_labels, batch_l), 0)

        data_train = joint_latent.get_lreval_data(data_train)

    return all_labels, data_train


def get_random_labels(n_samples, n_train_samples, all_labels, max_tries=1000):
    """
    The classifier needs labels from both classes to train. This function resamples "all_labels"
    until it contains examples from both classes
    """
    assert any(len(np.unique(all_labels[:, l])) > 1 for l in range(all_labels.shape[-1])), \
        'The labels must contain at least two classes to train the classifier'
    rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
    labels = all_labels[rand_ind_train, :]
    tries = 1
    while any(len(np.unique(labels[:, l])) <= 1 for l in range(labels.shape[-1])):
        rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
        labels = all_labels[rand_ind_train, :]
        tries += 1
        assert max_tries >= tries, f'Could not get sample containing both classes to train ' \
                                   f'the classifier in {tries} tries. Might need to increase batch_size'
    return labels, rand_ind_train


def test_clf_lr_all_subsets(clf_lr: Mapping[str, Mapping[str, LogisticRegression]], exp, which_lr: str):
    """
    Test the classifiers that were trained on latent representations.

    which_lr: either q0.mu or zk.
    """
    args = exp.flags
    mm_vae = exp.mm_vae
    mm_vae.eval()
    subsets = [*exp.subsets, 'joint']

    d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size, shuffle=False,
                          num_workers=exp.flags.dataloader_workers, drop_last=False)

    training_steps = exp.flags.steps_per_training_epoch or len(d_loader)
    log.info(f'Creating {training_steps} batches of latent representations for classifier testing '
             f'with a batch_size of {exp.flags.batch_size}.')

    clf_predictions = init_clf_predictions(subsets, which_lr, mm_vae)

    batch_labels = torch.Tensor()

    for iteration, (batch_d, batch_l) in enumerate(d_loader):
        if iteration > training_steps:
            break
        batch_labels = torch.cat((batch_labels, batch_l), 0)

        batch_d = dict_to_device(batch_d, exp.flags.device)

        _, joint_latent = mm_vae.module.inference(batch_d) if args.distributed else mm_vae.inference(batch_d)
        lr_subsets = joint_latent.subsets

        data_test = {key: getattr(joint_latent, f'get_{which_lr}')(key).cpu().data.numpy() for key in clf_predictions}

        clf_predictions_batch = classify_latent_representations(exp, clf_lr, data_test)
        clf_predictions_batch: Mapping[str, Mapping[str, np.array]]

        for subset in data_test:
            clf_predictions_batch_subset = torch.cat(tuple(
                torch.tensor(clf_predictions_batch[label][subset]).unsqueeze(1) for label in
                exp.labels), 1)

            clf_predictions[subset] = torch.cat([clf_predictions[subset], clf_predictions_batch_subset], 0)

    batch_labels = atleast_2d(batch_labels, -1)
    results = {}
    for subset in clf_predictions:
        # calculate metrics
        metrics = exp.metrics(clf_predictions[subset], batch_labels, str_labels=exp.labels)
        metrics_dict = metrics.evaluate()
        results[subset] = metrics.extract_values(metrics_dict)
    log.info(f'Lr eval results: {results}')

    return results


def init_clf_predictions(subsets, which_lr, mm_vae):
    # flow of joint methods only have the joint subset as zk.
    if which_lr == 'zk' and isinstance(mm_vae, FlowOfJointVAE):
        return {'joint': torch.Tensor()}
    else:
        return {subset: torch.Tensor() for subset in subsets}


def classify_latent_representations(exp, clf_lr: Mapping[str, Mapping[str, LogisticRegression]], data) \
        -> Mapping[str, Mapping[str, np.array]]:
    """
    Returns the classification of each subset of the powerset for each label.
    """
    clf_predictions = {}
    for label_str in exp.labels:
        clf_pred_subset = {}

        for s_key, data_rep in data.items():
            # get the classifier for the subset
            clf_lr_rep = clf_lr[label_str][s_key]

            clf_pred_subset[s_key] = clf_lr_rep.predict(data_rep)

        clf_predictions[label_str] = clf_pred_subset
    return clf_predictions


def train_clf_lr(exp, data, labels):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)))
    clf_lr_labels = {}
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l]
        clf_lr_reps = {}
        for s_key in data.keys():
            data_rep = data[s_key]
            if data_rep is not None:
                clf_lr_s = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
                if exp.flags.dataset == 'testing':
                    # when using the testing dataset, the vae data_rep might contain nans. Replace them for testing purposes
                    clf_lr_s.fit(np.nan_to_num(data_rep), gt)
                else:
                    clf_lr_s.fit(data_rep, gt)
                clf_lr_reps[s_key] = clf_lr_s
        clf_lr_labels[label_str] = clf_lr_reps
    return clf_lr_labels
