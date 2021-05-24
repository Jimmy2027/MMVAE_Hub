# -*- coding: utf-8 -*-
"""
Execute jupyter notebooks in experiments dir.
Run with "python run_notebook run-one my/experiments/path"
Or
"python run_notebook run-multiple my/experiments/path"
"""
from pathlib import Path

import ppb
import typer

from mmvae_hub import log
from mmvae_hub.base.utils.MongoDB import MongoDatabase
from mmvae_hub.experiment_vis.utils import run_notebook_convert

app = typer.Typer()


@app.command()
def run_one(exp_dir: str):
    exp_dir = Path(exp_dir).expanduser()
    log.info(f'Starting execution of experiment vis for experiment {exp_dir.name}')
    pdf_path = run_notebook_convert(exp_dir)
    expvis_url = ppb.upload(pdf_path, plain=True)
    log.info(f'Uploaded experiment vis to {expvis_url}')
    db = MongoDatabase(training=False, _id=exp_dir.name)
    db.insert_dict({'expvis_url': expvis_url})


@app.command()
def run_multiple(experiments_dir: str, run_all: bool = False):
    experiments_dir = Path(experiments_dir).expanduser()
    for exp_dir in experiments_dir.iterdir():
        if (exp_dir / 'experiment_vis.nbconvert.pdf').exists() or run_all:
            try:
                run_one(exp_dir=str(exp_dir))
            except Exception as e:
                print(e)
            # pdf_path = BaseTrainer.run_notebook_convert(exp_dir)
            # expvis_url = ppb.upload(pdf_path, plain=True)
            # db = MongoDatabase(training=False, _id=exp_dir.name)
            # db.insert_dict({'expvis_url': expvis_url})


if __name__ == '__main__':
    app()
