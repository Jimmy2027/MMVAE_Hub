# -*- coding: utf-8 -*-
"""
Upload zipped experiment folders to the mongo database.
Use with python upload_experimentzip.py upload_all --src_dir my_src_dir
"""
import glob
import shutil
import tempfile
import zipfile
from pathlib import Path

import ppb
import torch
import typer
from norby import send_msg

from mmvae_hub import log
from mmvae_hub.experiment_vis.utils import run_notebook_convert
from mmvae_hub.utils.MongoDB import MongoDatabase
from mmvae_hub.utils.utils import json2dict

app = typer.Typer()


def upload_one(exp_path: Path):
    """
    Upload one experiment result to database together with the model checkpoints,
    the logfile and tensorboardlogs, then delete zipped experiment dir.
    """
    is_zip = exp_path.suffix == '.zip'
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname) / exp_path.stem
        tmpdir.mkdir()

        if is_zip:
            # unpack zip into tmpdir
            log.info(f'Unpacking {exp_path} to {tmpdir}.')
            with zipfile.ZipFile(exp_path) as z:
                z.extractall(tmpdir)
            exp_dir = Path(tmpdir)
        else:
            exp_dir = exp_path

        flags = torch.load(exp_dir / 'flags.rar')
        db = MongoDatabase(training=True, flags=flags)
        results = {'epoch_results': {}}

        epochs = sorted(int(str(epoch.stem)) for epoch in (exp_dir / 'epoch_results').iterdir())

        for epoch in epochs:
            epoch_str = str(epoch)
            epoch_results = (exp_dir / 'epoch_results' / epoch_str).with_suffix('.json')
            results['epoch_results'][epoch_str] = json2dict(epoch_results)

        db.insert_dict(results)

        modalities = [mod_str for mod_str in results['epoch_results'][str(epoch)]['train_results']['log_probs']]
        dir_checkpoints = exp_dir / 'checkpoints'
        db.save_networks_to_db(
            dir_checkpoints=dir_checkpoints,
            epoch=max(int(str(d.name)) for d in dir_checkpoints.iterdir()),
            modalities=modalities,
        )

        pdf_path = run_notebook_convert(exp_dir)
        expvis_url = ppb.upload(pdf_path, plain=True)
        db.insert_dict({'expvis_url': expvis_url})

        log_file = glob.glob(str(exp_dir) + '/*.log')
        if len(log_file):
            db.upload_logfile(Path(log_file[0]))

        db.upload_tensorbardlogs(exp_dir / 'logs')

        send_msg(f'Uploading of experiment {flags.experiment_uid} has finished. The experiment visualisation can be '
                 f'found here: {expvis_url}')

    # delete exp_path
    if is_zip:
        exp_path.unlink()
    else:
        shutil.rmtree(exp_path)


@app.command()
def upload_all(src_dir: str):
    """
    Upload all experiment results to database together with the model checkpoints,
    the logfile and tensorboardlogs, then delete zipped experiment dir.
    """

    src_dir = Path(src_dir).expanduser()
    for experiment_zip in src_dir.iterdir():
        try:
            upload_one(experiment_zip)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # app()
    from norby.utils import norby

    # upload_one(Path('/Users/Hendrik/Documents/temp/Mimic_joint_elbo_2021_07_06_09_44_52_871882'))

    with norby('beginning upload experimentzip', 'finished beginning upload experimentmentzip'):
        # upload_all('/mnt/data/hendrik/leomed_results')
        upload_all('/Users/Hendrik/Documents/master_4/leomed_experiments')
