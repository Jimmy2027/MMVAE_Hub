import glob
import json
import os
import shutil
from pathlib import Path

from mmvae_hub.base.utils.MongoDB import MongoDatabase
from mmvae_hub.base.utils.flags_utils import get_config_path


def clean_database():
    """Delete all experiment logs in database from experiments with less than 10 epochs."""
    db = MongoDatabase(training=False)
    experiments = db.connect()

    for experiment in experiments.find({}):
        if experiment['flags']['end_epoch'] < 10 or len(experiment['epoch_results']) < 10:
            print(f'Deleting experiment {experiment["_id"]} from database.')
            experiments.delete_one({'_id': experiment['_id']})


def clean_database_model_checkpoints():
    db = MongoDatabase(training=False)
    experiments = db.connect()
    experiment_ids = [exp['_id'] for exp in experiments.find({})]
    print(experiment_ids)
    fs = db.connect_with_gridfs()

    for checkpoint in fs.find({}):
        checkpoint_exp_id = checkpoint._id.__str__().split('__')[0]
        if checkpoint_exp_id not in experiment_ids:
            print(f'Removing checkpoint {checkpoint._id}.')
            fs.delete(checkpoint._id)


def clean_exp_dirs(config: dict):
    """
    Removes all experiment dirs that don't have a log dir or where the log dir is empty.
    Theses experiment dir are rests from an unsuccessful "rm -r" command.
    """
    db = MongoDatabase(training=False)
    experiments = db.connect()
    experiment_ids = [exp['_id'] for exp in experiments.find({})]

    checkpoint_path = Path(config['dir_experiment']).expanduser()

    for experiment_dir in checkpoint_path.iterdir():
        if experiment_dir.name != 'fid' and experiment_dir.is_dir():
            remove = False
            d = experiments.find_one({'_id': experiment_dir.name})

            if (
                    not os.path.exists(os.path.join(experiment_dir, 'logs'))
                    or len(os.listdir(os.path.join(experiment_dir, 'logs'))) == 0
            ):
                remove = True


            elif experiment_dir.name not in experiment_ids:
                remove = True

            elif d['flags']['end_epoch'] < 10:
                remove = True

            if remove:
                print(f'removing dir {experiment_dir}')
                shutil.rmtree(experiment_dir)

            # elif not (experiment_dir / 'checkpoints').exists() or len(
            #         (experiment_dir / 'checkpoints').iterdir()) == 0:
            #     print(f'removing dir {experiment_dir}')
            #     shutil.rmtree(experiment_dir)
            #
            # elif (max(int(d.name) for d in (experiment_dir / 'checkpoints').iterdir() if d.name.startswith('0')) < 10):
            #     print(f'removing dir {experiment_dir}')
            #     shutil.rmtree(experiment_dir)


def clean_early_checkpoints(parent_folder: Path):
    for experiment_dir in parent_folder.iterdir():
        checkpoints_dir = parent_folder / experiment_dir / 'checkpoints/0*'
        checkpoints = glob.glob(checkpoints_dir.__str__())
        checkpoint_epochs = sorted([Path(checkpoint).stem for checkpoint in checkpoints])
        for checkpoint in checkpoints:
            if Path(checkpoint).stem != checkpoint_epochs[-1]:
                shutil.rmtree(checkpoint)


if __name__ == '__main__':
    clean_database()
    clean_database_model_checkpoints()
    config_path = get_config_path()
    with open(config_path, 'rt') as json_file:
        config = json.load(json_file)

    clean_exp_dirs(config)
