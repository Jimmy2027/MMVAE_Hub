# -*- coding: utf-8 -*-
"""Basic tests for the MMVAE_mst package."""

from pathlib import Path

import pytest
from pymongo import MongoClient

from mmvae_hub.utils.utils import json2dict


@pytest.mark.tox
def test_import():
    """Tests the imports for the MMVAE_mst package."""
    import mmvae_hub
    print(mmvae_hub.__file__)


def test_connectdb():
    """Test connection to the mongodb database."""
    dbconfig = json2dict(Path(__file__).parent.parent / 'configs/mmvae_db.json')
    client = MongoClient(dbconfig['mongodb_URI'])
    db = client.mmvae
    experiments = db.experiments
    print(experiments.find_one)


if __name__ == '__main__':
    test_connectdb()
