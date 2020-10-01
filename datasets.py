"""Module contains functions for working with datasets."""
import os
import re

import dedalus.public as de
import h5py
import numpy as np
import pandas as pd


def select_measurements_from_dedalus_sim(
        xi, yi, sim_dir, stem='analysis_tasks'
):
    """Select measurements from a Dedalus simulation.

    Parameters
    ----------
    xi, yi : ndarray of dtype int
        Indices of the grid points in x- and y-direction.
    sim_dir : str
        Directory, in which simulation results are stored.
    stem : str, optional (default 'analysis_tasks')
        Subdirectory of `sim_dir`, in which HDF5 files of simulations are
        written.

    Returns
    -------
    df : pd.Dataframe
        Pandas dataframe with columns `time`, `location_id`, `u`, `v`.
    """
    all_files = os.listdir(os.path.join(sim_dir, stem))
    pattern = stem + '_s[0-9]+.h5'
    sim_filenames = sorted([f for f in all_files if re.match(pattern, f)])

    sample_t, sample_loc, sample_u, sample_v = [], [], [], []

    for sf in sim_filenames:
        fn = os.path.join(sim_dir, stem, sf)
        fh = h5py.File(fn, 'r')

        for (i, (x, y)) in enumerate(zip(xi, yi)):
            u = fh['tasks']['u'][:, x, y]
            v = fh['tasks']['v'][:, x, y]
            sim_times = fh['scales']['sim_time'][...]
            nsim_times = len(sim_times)
            sample_u.extend(u)
            sample_v.extend(v)
            sample_t.extend(sim_times)
            sample_loc.extend(nsim_times * [i])

    df = pd.DataFrame({
        "time": sample_t,
        "location_id": sample_loc,
        "u": sample_u,
        "v": sample_v,
    })

    return df
