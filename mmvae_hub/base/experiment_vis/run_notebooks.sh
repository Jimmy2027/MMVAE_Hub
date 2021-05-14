#!/usr/bin/env bash

EXPERIMENTS_DIR="${1}"


for experiment_dir in $EXPERIMENTS_DIR/*
do
#  if [ -f "$experiment_dir/experiment_vis.ipynb" ]; then
  if true; then
#    if [ -f "$experiment_dir/experiment_vis.nbconvert.html" ]; then
    if false; then
      echo "$experiment_dir/experiment_vis.nbconvert.html exists."
    else
      echo "$experiment_dir/experiment_vis.nbconvert.html does not exist."
      cp experiment_vis.ipynb "$experiment_dir/experiment_vis.ipynb"

      jupyter nbconvert --to notebook --execute "$experiment_dir"/experiment_vis.ipynb --allow-errors || exit 1
      jupyter nbconvert --to html "$experiment_dir"/experiment_vis.nbconvert.ipynb || exit 1
    fi
  fi
done
