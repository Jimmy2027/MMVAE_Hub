# -*- coding: utf-8 -*-
import pathlib
from pathlib import Path

import torch
from pymongo import MongoClient

from mmvae_hub import log
from mmvae_hub.base.utils.utils import json2dict


class MongoDatabase:
    def __init__(self, flags, training: bool = True):
        """
        training: if true, experiment_uid and flags will be send to db.
        """
        self.mongodb_URI = self.get_mongodb_uri()
        self.experiment_uid = flags.experiment_uid

        # create document in db for current experiment
        log.info('Connecting to database.')
        experiments = self.connect()
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


if __name__ == '__main__':
    from dataclasses import dataclass


    @dataclass()
    class DC:
        experiment_uid: str = 'temp_uid'


    dc = DC()

    db = MongoDatabase(dc, training=False)
    db.delete_all()
