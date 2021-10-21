"""Delete all checkpoints before a certain epoch and at a certain interval."""

import tempfile
from pathlib import Path

from modun.file_io import json2dict
from modun.zip_utils import unzip_to
import shutil

from mmvae_hub.utils.setup.flags_utils import get_config_path


def cleanup(checkpoints_dir: Path):
    checkpoint_paths = list(checkpoints_dir.iterdir())
    last_checkpoint_epoch = max(
        int(checkpoint_path.stem) for checkpoint_path in checkpoint_paths
    )

    # delete checkpoints
    for checkpoint_path in checkpoint_paths:
        epoch = int(checkpoint_path.stem)
        if (
                epoch != last_checkpoint_epoch
                and (epoch <= 450 or epoch % 100 != 99)
        ):
            shutil.rmtree(checkpoint_path)


if __name__ == '__main__':
    # polymnist_moe_2021_09_30_09_21_13_361319
    # config = json2dict(get_config_path(dataset='polymnist'))

    # dir_experiments = Path(config['dir_experiment']).expanduser()

    dir_experiments = Path('/Users/Hendrik/Documents/master_4/MMNF_RepSeP/data/thesis/experiments/mopoe')

    for exp_dir in dir_experiments.iterdir():

        if exp_dir.suffix != '.zip':
            print(f'{exp_dir} is dir')
            # cleanup(exp_dir / 'checkpoints')

        else:
            # unzip to tempdir
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_exp_dir = Path(tmpdirname) / exp_dir.stem
                unzip_to(dest_path=temp_exp_dir, verbose=True, path_to_zip_file=exp_dir)

                if not temp_exp_dir.exists():
                    print(f'{temp_exp_dir} not in parent directory: {list(temp_exp_dir.parent.iterdir())}')
                elif not (temp_exp_dir / 'checkpoints').exists():
                    print(
                        f'{temp_exp_dir / "checkpoints"} not in parent directory: {list((temp_exp_dir / "checkpoints").parent.iterdir())}')
                else:
                    try:
                        cleanup((temp_exp_dir / 'checkpoints'))
                        failed = False
                    except Exception as e:
                        print(e)
                        failed = True

                    if not failed:
                        assert (Path(tmpdirname) / exp_dir.stem).exists()

                    # delete old zip file
                    exp_dir.unlink()

                    # re-zip to original place
                    shutil.make_archive(str(exp_dir.parent / exp_dir.stem), 'zip', str(Path(tmpdirname) / exp_dir.stem),
                                        verbose=True)
