# -*- coding: utf-8 -*-
"""Upload zipped experiment folders to the mongo database."""

import tempfile
import zipfile
from pathlib import Path

import typer

from mmvae_hub.base.utils.MongoDB import MongoDatabase
from mmvae_hub.base.utils.utils import json2dict
import torch

app = typer.Typer()


@app.command()
def upload_all(src_dir: str):
    """Unzip experiment_dir to a tmpdir, upload all experiment results to database together with the model checkpoints, the logfile and tensrboardlogs, then delete zipped experiment dir."""

    src_dir = Path(src_dir).expanduser()
    for experiment_zip in src_dir.iterdir():
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(experiment_zip) as z:
                z.extractall(tmpdirname)
            exp_dir = Path(tmpdirname)
            flags = torch.load(exp_dir / 'flags.rar')
            db = MongoDatabase(training=True, flags=flags)

            for epoch_results in Path(tmpdirname).iterdir():
                epoch_results_dict = json2dict(epoch_results)
                db.insert_dict({'epoch_results': epoch_results_dict})

            # read the modality strs from results.json
            modalities = [mod_str for mod_str in json2dict(exp_dir / 'results.json')['log_probs']]
            dir_checkpoints = exp_dir / 'checkpoints'
            db.save_networks_to_db(
                dir_checkpoints=dir_checkpoints,
                epoch=max(int(str(d)) for d in dir_checkpoints.iterdir()),
                modalities=modalities,
            )


    log.info(f'Starting execution of experiment vis for experiment {exp_dir.name}')
    pdf_path = BaseTrainer.run_notebook_convert(exp_dir)
    expvis_url = ppb.upload(pdf_path, plain=True)
    log.info(f'Uploaded experiment vis to {expvis_url}')
    db = MongoDatabase(training=False, _id=exp_dir.name)
    db.insert_dict({'expvis_url': expvis_url})


if __name__ == '__main__':
    app()
