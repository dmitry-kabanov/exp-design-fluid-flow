#!/usr/bin/env python
"""Choose sparse measurements from a given Dedalus simulation."""
import os

import datasets

SIM_DIR = os.path.join('_output', 'KH_1e5_a01')
# INPUT_DIR = os.path.join('_output', 'KH_1e5_a01_noisy')

# Number of measurements to select from the full solution.
N = 100

# Random seed to choose random measurements deterministically.
random_seed = 42

# The same name that was used for the file handler of the simulation writer.
stem = 'analysis_tasks'

t, x, y, u, v = datasets.select_measurements_from_dedalus_sim(
    SIM_DIR, N, random_seed=random_seed, stem=stem
)
