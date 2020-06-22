#!/usr/bin/env python
# coding: utf-8


import subprocess
import numpy as np

#ds=[2, 3, 4]
ds=np.concatenate([np.arange(2, 11), np.arange(12, 38, 2), np.arange(40, 100, 4), [128]])
seeds=np.arange(10)
nb_exps=1000

for d in ds:
    for seed in seeds:
        subprocess.call(['bsub', '-W 3:50', '-n 4', '-R', "rusage[mem=2048]", "python mcmc_gaussian_nonisotropic_d_fixedkappa_exps.py %d %d %d" %(d, seed, nb_exps)])
