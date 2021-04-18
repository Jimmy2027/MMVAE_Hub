# MMVAE_Hub
[![Build Status](https://travis-ci.com/Jimmy2027/MMVAE_mnist_svhn_text.svg?branch=main)](https://travis-ci.com/Jimmy2027/MMVAE_mnist-svhn-text)

Codebase for training multi modal VAEs on multiple datasets.

## Installation
This repository depends on the codebase from [MMVAE_base](https://github.com/Jimmy2027/MMVAE_base).
```
path/to/conda/environment/bin/python -m pip install git+https://github.com/Jimmy2027/MMVAE_base
path/to/conda/environment/bin/python -m pip install git+https://github.com/Jimmy2027/MMVAE_Hub
```

For development, install with: 
```
git clone git@github.com:Jimmy2027/MMVAE_Hub.git
cd MMVAE_Hub
path/to/conda/environment/bin/python -m pip install -e .

git clone git@github.com:Jimmy2027/MMVAE_base.git
cd MMVAE_base
path/to/conda/environment/bin/python -m pip install -e .
```