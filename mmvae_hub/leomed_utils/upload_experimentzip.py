# -*- coding: utf-8 -*-
"""
Upload zipped experiment folders to the mongo database.
Use with python upload_experimentzip.py upload_all --src_dir my_src_dir
"""
import shutil
import glob
import tempfile
import zipfile
from pathlib import Path

import ppb
import torch
import typer

from mmvae_hub.base.BaseTrainer import BaseTrainer
from mmvae_hub.utils import MongoDatabase
from mmvae_hub.utils import json2dict

app = typer.Typer()


@app.command()
def upload_all(src_dir: str, is_zip: bool = True):
    """
    If is_zip is True, unzip experiment_dir to a tmpdir.
    Upload all experiment results to database together with the model checkpoints,
    the logfile and tensrboardlogs, then delete zipped experiment dir.
    """

    src_dir = Path(src_dir).expanduser()
    for experiment_zip in src_dir.iterdir():
        with tempfile.TemporaryDirectory() as tmpdirname:
            if is_zip:
                # unpack zip into tmpdir
                with zipfile.ZipFile(experiment_zip) as z:
                    z.extractall(tmpdirname)
                exp_dir = Path(tmpdirname)
            else:
                exp_dir = experiment_zip

            flags = torch.load(exp_dir / 'flags.rar')
            db = MongoDatabase(training=True, flags=flags)

            for epoch_results in Path(exp_dir / 'epoch_results').iterdir():
                epoch = epoch_results.stem
                epoch_results_dict = json2dict(epoch_results)
                epoch_results = db.get_experiment_dict()['epoch_results']
                epoch_results[f'{epoch}'] = epoch_results_dict
                db.insert_dict({'epoch_results': epoch_results})

            # read the modality strs from results.json
            modalities = [mod_str for mod_str in epoch_results_dict['train_results']['log_probs']]
            dir_checkpoints = exp_dir / 'checkpoints'
            db.save_networks_to_db(
                dir_checkpoints=dir_checkpoints,
                epoch=max(int(str(d.name)) for d in dir_checkpoints.iterdir()),
                modalities=modalities,
            )

            log_file = glob.glob(str(exp_dir) + '/*.log')
            if len(log_file):
                db.upload_logfile(Path(log_file[0]))
            db.upload_tensorbardlogs(exp_dir / 'logs')

            pdf_path = BaseTrainer.run_notebook_convert(exp_dir)
            expvis_url = ppb.upload(pdf_path, plain=True)
            db.insert_dict({'expvis_url': expvis_url})

        # delete experiment_zip
        shutil.rmtree(experiment_zip)


if __name__ == '__main__':
    # app()
    # upload_all('/Users/Hendrik/Documents/master_4/leomed_experiments')
    upload_all('/mnt/data/hendrik/leomed_results/', is_zip=False)
