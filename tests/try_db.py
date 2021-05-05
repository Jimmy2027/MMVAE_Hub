# -*- coding: utf-8 -*-
import datetime
from pathlib import Path

from pymongo import MongoClient

from mmvae_hub.base.utils.utils import json2dict

dbconfig = json2dict(Path(__file__).parent.parent / 'configs/mmvae_db.json')
client = MongoClient(dbconfig['mongodb_URI'])
db = client.mmvae
posts = db.posts

post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}

id = posts.find_one()['_id']

db.posts.find_one_and_update({'_id': id}, {"$set": {'bruh': 'yeah'}})
