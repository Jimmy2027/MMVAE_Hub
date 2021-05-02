import glob
import json
import os
import shutil
from pathlib import Path

from mmvae_hub.base.utils.flags_utils import get_config_path


def clean_exp_dirs(config: dict):
    """
    Removes all experiment dirs that don't have a log dir or where the log dir is empty.
    Theses experiment dir are rests from an unsuccessful "rm -r" command.
    """

    checkpoint_path = Path(config['dir_experiment']).expanduser()

    for experiment_dir in checkpoint_path.iterdir():
        if experiment_dir.is_dir():
            if (
                    not os.path.exists(os.path.join(experiment_dir, 'logs'))
                    or len(os.listdir(os.path.join(experiment_dir, 'logs'))) == 0
            ):
                print(f'removing dir {experiment_dir}')
                shutil.rmtree(experiment_dir)
            elif len(
                    os.listdir(os.path.join(experiment_dir, 'checkpoints'))) == 0:
                print(f'removing dir {experiment_dir}')
                shutil.rmtree(experiment_dir)

            elif (max(int(d.name) for d in (experiment_dir / 'checkpoints').iterdir() if d.name.startswith('0')) < 10):
                print(f'removing dir {experiment_dir}')
                shutil.rmtree(experiment_dir)


def clean_early_checkpoints(parent_folder: Path):
    for experiment_dir in parent_folder.iterdir():
        checkpoints_dir = parent_folder / experiment_dir / 'checkpoints/0*'
        checkpoints = glob.glob(checkpoints_dir.__str__())
        checkpoint_epochs = sorted([Path(checkpoint).stem for checkpoint in checkpoints])
        for checkpoint in checkpoints:
            if Path(checkpoint).stem != checkpoint_epochs[-1]:
                shutil.rmtree(checkpoint)


if __name__ == '__main__':
    config_path = get_config_path()
    with open(config_path, 'rt') as json_file:
        config = json.load(json_file)

    clean_exp_dirs(config)
