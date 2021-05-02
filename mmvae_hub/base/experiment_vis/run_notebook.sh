#!/usr/bin/env bash

jupyter nbconvert --to notebook --execute experiment_vis.ipynb && jupyter nbconvert --to html experiment_vis.nbconvert.ipynb

cp experiment_vis.nbconvert.html /mnt/data/hendrik/mmvae_hub/experiments/polymnist_planar_mixture_2021_04_29_23_06_00_937191/