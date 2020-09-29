"""Module contains functions for working with datasets."""
import os
import re

import dedalus.public as de
import h5py
import numpy as np


def select_measurements_from_dedalus_sim(
        sim_dir, N, random_seed=42, stem='analysis_tasks'
):
    """Select measurements from a Dedalus simulation.

    Parameters
    ----------
    sim_dir : str
        Directory, in which simulation results are stored.
    N : int
        Number of measurements to select.
    random_seed : int, optional (default 42)
        Random seed for selection of indices.
    stem : str, optional (default 'analysis_tasks')
        Subdirectory of `sim_dir`, in which HDF5 files of simulations are
        written.

    Returns
    -------
    sample_t, sample_x, sample_y, sample_u, sample_v : ndarray
        Tuple of selected measurements: time, x-coordinate, y-coordinate,
        x-component of velocity, y-component of velocity.
    """
    np.random.seed(random_seed)

    all_files = os.listdir(os.path.join(sim_dir, stem))
    pattern = stem + '_s[0-9]+.h5'
    sim_filenames = [f for f in all_files if re.match(pattern, f)]
    sim_filenames = sorted(sim_filenames)

    sim_fh = []
    sim_times = []
    sim_ts_to_fh = []
    sim_ts_to_offset = []
    ti = 0
    for sf in sim_filenames:
        fn = os.path.join(sim_dir, stem, sf)
        fh = h5py.File(fn, 'r')
        sim_fh.append(fh)
        nsim_times = len(fh['scales']['sim_time'][...])
        sim_times.extend(fh['scales']['sim_time'][...])
        sim_ts_to_fh.extend(nsim_times * [fh])
        sim_ts_to_offset.extend(nsim_times * [ti])
        ti = ti + nsim_times

    nt = len(sim_ts_to_fh)

    # Create bases and domain
    # TODO: This is ugly hardcode that must be removed by saving the grid
    # into output files.
    Lx = sim_fh[0]['tasks']['Lx'][0, 0, 0]
    Ly = sim_fh[0]['tasks']['Ly'][0, 0, 0]
    nx = int(sim_fh[0]['tasks']['nx'][0, 0, 0])
    ny = int(sim_fh[0]['tasks']['ny'][0, 0, 0])
    dealiasx = 3 / 2
    dealiasy = 3 / 2
    x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=dealiasx)
    y_basis = de.Chebyshev("y", ny, interval=(-Ly / 2, Ly / 2), dealias=dealiasy)
    x_grid = x_basis.grid()
    y_grid = y_basis.grid()

    # The total number of measurements is `NT \times NX \times NY`.
    N_TOTAL = nt * nx * ny

    # Choose uniform `N` measurements from `N_TOTAL` measurements.
    # First, generate numbers from 0 to `N_TOTAL` and then transform them
    # to the indices of the array.
    idx_flat = np.random.choice(N_TOTAL, N)
    idx = np.unravel_index(idx_flat, (nt, nx, ny))

    # Question: should components of u, v, and s be chosen with different indices?
    sample_t = []
    sample_x = []
    sample_y = []
    sample_u = []
    sample_v = []
    for i in range(N):
        ti = idx[0][i]
        fh = sim_ts_to_fh[ti]
        ti_offset = sim_ts_to_offset[ti]
        sample_t.append(sim_times[ti])
        sample_x.append(x_grid[idx[1][i]])
        sample_y.append(x_grid[idx[2][i]])
        sample_u.append(fh['tasks']['u'][ti-ti_offset, idx[1][i], idx[2][i]])
        sample_v.append(fh['tasks']['v'][ti-ti_offset, idx[1][i], idx[2][i]])
    sample_t = np.array(sample_t)
    sample_x = np.array(sample_x)
    sample_y = np.array(sample_y)
    sample_u = np.array(sample_u)
    sample_v = np.array(sample_v)

    return sample_t, sample_x, sample_y, sample_u, sample_v
