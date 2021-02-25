#!/bin/bash
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=jupyter --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-31 $@
