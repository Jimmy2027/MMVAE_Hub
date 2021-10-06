def dataset_to_metric(dataset: str) -> str:
    if dataset == 'celeba':
        return 'avg_prec'
    else:
        return 'accuracy'
