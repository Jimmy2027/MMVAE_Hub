# -*- coding: utf-8 -*-
import tempfile
import zipfile
from pathlib import Path

import wget


def download_polymnist_clfs(out_path: Path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_file = f'{tmpdirname}/{out_path.stem}.zip'
        url = "https://www.dropbox.com/sh/esbj7b6bg99jxna/AAAIa943_mueszIwh5721UMXa?dl=1"
        wget.download(out=zip_file, url=url)
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(out_path)

    assert out_path.exists()
