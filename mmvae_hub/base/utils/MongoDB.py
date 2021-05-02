# -*- coding: utf-8 -*-
import pathlib
from pathlib import Path

import torch
from pymongo import MongoClient

from mmvae_hub import log
from mmvae_hub.base.utils.utils import json2dict


class MongoDatabase:
    def __init__(self, flags):
        self.mongodb_URI = self.get_mongodb_uri()
        self.experiment_uid = flags.experiment_uid

        # create document in db for current experiment
        log.info('Connecting to database.')
        client = MongoClient(self.mongodb_URI)
        db = client.mmvae
        experiments = db.experiments
        if self.experiment_uid not in [str(id) for id in experiments.find().distinct('_id')]:
            experiments.insert_one({'_id': self.experiment_uid, 'flags': self.encode_flags(flags)})

    @staticmethod
    def get_mongodb_uri():
        dbconfig = json2dict(Path(__file__).parent.parent.parent.parent / 'configs/mmvae_db.json')
        return dbconfig['mongodb_URI']

    def insert_dict(self, d: dict):
        log.info('Inserting dict to database.')
        client = MongoClient(self.mongodb_URI)
        db = client.mmvae
        experiments = db.experiments
        experiments.find_one_and_update({'_id': self.experiment_uid}, {"$set": d})

    @staticmethod
    def encode_flags(flags):
        flags_dict = vars(flags).copy()
        for k, elem in flags_dict.items():
            if type(elem) in [pathlib.PosixPath, torch.device]:
                flags_dict[k] = str(elem)

        return flags_dict
