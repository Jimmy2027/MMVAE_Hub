# MMVAE_Hub
[![Build Status](https://travis-ci.com/Jimmy2027/MMVAE_Hub.svg?branch=main)](https://travis-ci.com/Jimmy2027/MMVAE_Hub)

Codebase for training multi modal VAEs on multiple datasets.

## Installation

```
path/to/conda/environment/bin/python -m pip install git+https://github.com/Jimmy2027/MMVAE_Hub
```

For development, install with: 
```
git clone git@github.com:Jimmy2027/MMVAE_Hub.git
cd MMVAE_Hub
path/to/conda/environment/bin/python -m pip install -e .

```
## Usage
For each dataset, a config file exists under `configs/{dataset}`. The `local_config.json` is chosen per default (see `utils.setup.flags_utils.get_config_path`).
Experiments for each dataset (celeba, polymnist, mnistsvhntext, mimic) can be launched with:
````
python {dataset}/main_{dataset}.py
````

## Working on leomed
When working on leomed, please set the flag "leomed" to true.
On leomed, several steps are taken to reduce the number of files:
- the dataset is stored as zipfile, and will be unzipped in the $TMPDIR during runtime.
- the `experiment_dir` will also be set to $TMPDIR during runtime. At the end of the run, this directory will be 
  compressed to a zip file and stored in the `experiment_dir` defined in the config. 