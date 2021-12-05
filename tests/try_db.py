# -*- coding: utf-8 -*-
import datetime
import json
from pathlib import Path

from pymongo import MongoClient

json_path = Path(__file__).parent.parent / 'configs/mmvae_db.json'
with open(json_path, 'rt') as json_file:
    json_config = json.load(json_file)
client = MongoClient(json_config['mongodb_URI'])
db = client.mmvae
posts = db.posts

post = {"author": "Mike",
        "text": "My first blog post!",
        "tags": ["mongodb", "python", "pymongo"],
        "date": datetime.datetime.utcnow()}

id = posts.find_one()['_id']

db.posts.find_one_and_update({'_id': id}, {"$set": {'bruh': 'yeah'}})
