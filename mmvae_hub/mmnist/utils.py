# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path
import wget

def download_polymnist_clfs(out_path: Path):
    url = "https://www.dropbox.com/sh/esbj7b6bg99jxna/AAAIa943_mueszIwh5721UMXa?dl=1"
    with tempfile.TemporaryDirectory() as tmpdirname:
        wget.download(out=f'{tmpdirname}/{out_path.stem}.zip', url=url)

        unzip_command = f'unzip {tmpdirname}/{out_path.stem + ".zip"} -d {out_path}/'
        os.system(unzip_command)

    assert out_path.exists()
