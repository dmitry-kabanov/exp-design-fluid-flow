#!/usr/bin/env python
"""Choose sparse measurements from a given Dedalus simulation."""
import os

import dedalus.public as de
import numpy as np

import datasets


# %% Setting variables.
SIM_DIR = os.path.join('_output', 'KH_1e5_a01')
# SIM_DIR = os.path.join('_output', 'KH_1e5_a01_noisy')

# Number of measurement location points.
N = 100

# Random seed to choose random measurements deterministically.
random_seed = 42

# The same name that was used for the file handler of the simulation writer.
stem = 'analysis_tasks'

# %% Create bases and domain
# TODO: This is ugly hardcode that must be removed by saving the grid
# into output files.
Lx = 2.0
Ly = 1.0
nx = 1024
ny = 512
dealiasx = 3 / 2
dealiasy = 3 / 2
x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=dealiasx)
y_basis = de.Chebyshev("y", ny, interval=(-Ly / 2, Ly / 2), dealias=dealiasy)
x_grid = x_basis.grid()
y_grid = y_basis.grid()

# %% Select uniformly at random indices of points along x- and y- axes.
np.random.seed(random_seed)
idx = np.random.choice(nx*ny, N)
idx = np.unravel_index(idx, (nx, ny))
xi, yi = idx
xp, yp = x_grid[xi], y_grid[yi]

# %% Select measurements finally.
df = datasets.select_measurements_from_dedalus_sim(
    xi, yi, SIM_DIR, stem=stem
)

# %% Resultant dataframe:
# In [37]: df
# Out[37]:
#             time    location_id           u           v
# 0       0.000000              0    0.500000   -0.020024
# 1       0.101121              0    0.477832   -0.091477
# ...          ...            ...         ...         ...
# 20099  20.000457             99   -0.716112    0.002324
# [20100 rows x 4 columns]

# %% Plotting measurement locations.
plt.figure()
plt.scatter(xp, yp)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout(pad=0.1)
plt.show()

# %% Plotting time series for location 0.
ts_loc_0 = df[df['location_id'] == 0]
plt.figure()
plt.plot(ts_loc_0['time'], ts_loc_0['u'], label='u')
plt.plot(ts_loc_0['time'], ts_loc_0['v'], '--', label='v')
plt.xlabel('time')
plt.ylabel('Time series of velocity component')
plt.legend()
plt.tight_layout(pad=0.1)
plt.show()
