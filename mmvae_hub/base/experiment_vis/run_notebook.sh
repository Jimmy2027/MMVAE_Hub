#!/usr/bin/env bash

EXPERIMENT_DIR="${1}"

cp experiment_vis.ipynb "$EXPERIMENT_DIR"/ || exit 1

jupyter nbconvert --to notebook --execute "$EXPERIMENT_DIR"/experiment_vis.ipynb || exit 1
jupyter nbconvert --to html "$EXPERIMENT_DIR"/experiment_vis.nbconvert.ipynb || exit 1
