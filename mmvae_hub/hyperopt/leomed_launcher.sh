#!/usr/bin/env bash

bsub -n 8 -W 10:00 -R rusage[mem=1500,ngpus_excl_p=1,scratch=1000] -R select[gpu_mtotal0>=5000] python hyperopt/HyperoptTrainer.py
