# -*- coding: utf-8 -*-
import glob
import io
import pathlib
import tempfile
from pathlib import Path

import gridfs
import torch
from pymongo import MongoClient

from mmvae_hub import log
from mmvae_hub.base import BaseMMVae
from mmvae_hub.base.utils.utils import json2dict


class MongoDatabase:
    def __init__(self, flags=None, training: bool = True, _id: str = None):
        """
        training: if true, experiment_uid and flags will be sent to db.
        """
        self.mongodb_URI = self.get_mongodb_uri()

        if flags is not None:
            self.experiment_uid = flags.experiment_uid
        elif _id is not None:
            self.experiment_uid = _id

        experiments = self.connect()
        # create document in db for current experiment
        if training and self.experiment_uid not in [str(id) for id in experiments.find().distinct('_id')]:
            experiments.insert_one({'_id': self.experiment_uid, 'flags': self.encode_flags(flags), 'epoch_results': {},
                                    'version': flags.version})

    def connect(self):
        client = MongoClient(self.mongodb_URI)
        db = client.mmvae
        return db.experiments

    @staticmethod
    def get_mongodb_uri():
        dbconfig = json2dict(Path(__file__).parent.parent.parent.parent / 'configs/mmvae_db.json')
        return dbconfig['mongodb_URI']

    def insert_dict(self, d: dict):
        log.info('Inserting dict to database.')
        experiments = self.connect()
        experiments.find_one_and_update({'_id': self.experiment_uid}, {"$set": d})

    @staticmethod
    def encode_flags(flags):
        flags_dict = vars(flags).copy()
        for k, elem in flags_dict.items():
            if type(elem) in [pathlib.PosixPath, torch.device]:
                flags_dict[k] = str(elem)

        return flags_dict

    def get_experiment_dict(self):
        experiments = self.connect()
        return experiments.find_one({'_id': self.experiment_uid})

    def delete_all(self):
        """
        Removes all documents in database.
        """
        experiment = self.connect()
        experiment.delete_many({})

    def delete_one(self, _id: str):
        """
        Removes one document from db
        """
        log.info(f'Deleting document with _id: {_id}.')
        experiment = self.connect()
        experiment.delete_one({'_id': _id})

    def save_networks_to_db(self, dir_checkpoints: Path, epoch: int, modalities):
        """
        Inspired from https://medium.com/naukri-engineering/way-to-store-large-deep-learning-models-in-production-ready-environments-d8a4c66cc04c
        There is probably a better way to store Tensors in MongoDB.
        """
        client = MongoClient(self.mongodb_URI)
        db = client.mmvae
        fs = gridfs.GridFS(db)
        checkpoint_dir = dir_checkpoints / str(epoch).zfill(4)

        for mod_str in modalities:
            for prefix in ['en', 'de']:
                filename = checkpoint_dir / f"{prefix}coderM{mod_str}"
                with io.FileIO(str(filename), 'r') as fileObject:
                    log.info(f'Saving checkpoint to db: {filename}')
                    fs.put(fileObject, filename=str(filename),
                           _id=self.experiment_uid + f"__{prefix}coderM{mod_str}")

    def connect_with_gridfs(self):
        client = MongoClient(self.mongodb_URI)
        db = client.mmvae
        return gridfs.GridFS(db)

    def load_networks_from_db(self, mmvae: BaseMMVae):
        fs = self.connect_with_gridfs()
        fs_ids = [elem._id for elem in fs.find({})]

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            for mod_str in mmvae.modalities:
                for prefix in ['en', 'de']:
                    filename = tmpdirname / f"{prefix}coderM{mod_str}"
                    model_id = self.experiment_uid + f"__{prefix}coderM{mod_str}"
                    with open(filename, 'wb') as fileobject:
                        fileobject.write(fs.get(model_id).read())

            mmvae.load_networks(tmpdirname)
        return mmvae

    def load_experiment_results_to_db(self, experiments_dir: Path):
        """Iterate through the experiment_dir and load results to the db if they are not already there."""
        experiments = self.connect()
        exp_ids_db = [elem['_id'] for elem in experiments.find({})]

        fs = self.connect_with_gridfs()
        fs_ids = {elem._id.split('__')[0] for elem in fs.find({})}

        for exp_dir in experiments_dir.iterdir():
            print(exp_dir)
            # get epoch results
            if exp_dir.name not in exp_ids_db:
                if (exp_dir / 'epoch_results').exists():
                    for epoch_result_dir in (exp_dir / 'epoch_results').iterdir():
                        # todo load epoch results to db
                        pass

            # get models
            if exp_dir.name not in fs_ids:
                if (exp_dir / 'checkpoints').exists():

                    self.experiment_uid = exp_dir.name

                    latest_checkpoint = max(
                        int(d.name) for d in (exp_dir / 'checkpoints').iterdir() if d.name.isdigit())
                    dir_checkpoints = (exp_dir / 'checkpoints' / str(latest_checkpoint).zfill(4))
                    modalities = {Path(e).name.replace('decoderM', '') for e in
                                  glob.glob(str(dir_checkpoints / 'decoderM*'))}
                    self.save_networks_to_db(
                        dir_checkpoints=(exp_dir / 'checkpoints'),
                        epoch=latest_checkpoint, modalities=modalities)
                else:
                    print('checkpoint dir does not exist')


if __name__ == '__main__':
    from dataclasses import dataclass


    @dataclass()
    class DC:
        experiment_uid: str = 'temp_uid'


    dc = DC()

    mongo_db = MongoDatabase(dc, training=False)
    mongo_db.delete_one('polymnist_planar_mixture_2021_05_04_09_21_38_767516')

    sdfg = 0
