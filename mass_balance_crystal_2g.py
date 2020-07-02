#%matplotlib inline
'''
Run the simulation on 4 processors
    $ mpiexec -n 4 python3 mass_balance_crystal_2g.py
Merge the results into a single analysis_tasks_s$.h5 is done at the end of this script, no need for
    $ mpiexec -n 4 python3 -m dedalus merge_procs analysis_tasks
Copy an analysis_tasks_s$.h5 file as restart.h5 in the folder /analysis_tasks to restart the simulation
    $ cd /analysis_tasks; cp analysis_tasks_s$.h5 restart.h5; cd ../
Run the simulation again to restart it
    $ mpiexec -n 4 python3 mass_balance_crystal_2g.py
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
import time
from IPython import display
import vtk_io as vtk
import pathlib

from chemical_potential_2D import mu_2g
from ic_crystal import ic

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

#Aspect ratio 2
Lx, Ly = (40.0*np.pi/np.sqrt(3.0), 24.0*np.pi)
nx, ny = (128, 92)
dim = 2

# Create bases and domain
dealiasx = 3/2
dealiasy = 3/2
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=dealiasx)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=dealiasy)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

Lp = 1.0
La = 1.0
Lb = 1.0
Lc = 1.0

problem = de.IVP(domain, variables=['s','sy','mu','muy'])
problem.parameters['Lp'] = Lp
problem.parameters['La'] = La
problem.parameters['Lb'] = Lb
problem.parameters['Lc'] = Lc
problem.parameters['nx'] = nx
problem.parameters['ny'] = ny
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['dim'] = dim
problem.add_equation("dt(s) - 1/Lp*(dx(dx(mu)) + dy(muy)) = 0")
problem.add_equation("muy - dy(mu) = 0")
problem.add_equation("sy - dy(s) = 0")
problem.add_equation(mu_2g(La,Lb,Lc))

ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

# Initial conditions or restart
print(pathlib.Path)
if not pathlib.Path('./analysis_tasks/restart.h5').exists():

    x = domain.grid(0)
    y = domain.grid(1)
    s = solver.state['s']
    sy = solver.state['sy']
    mu = solver.state['mu']
    muy = solver.state['muy']

    s['g'] = ic(x,y)
    s.differentiate('y',out=sy)

    stop_sim_time  = 1.0e-3
    stop_wall_time = np.inf
    stop_iteration = np.inf

    initial_dt = 1.0e-5
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('./analysis_tasks/restart.h5', -1)

    # Timestepping and output
    print(last_dt)
    initial_dt = last_dt
    stop_sim_time  = 5.01
    stop_wall_time = np.inf
    stop_iteration = np.inf
    fh_mode = 'append'
    s = solver.state['s']

solver.stop_sim_time  = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

analysis = solver.evaluator.add_file_handler('analysis_tasks', iter=100, max_writes=200, mode=fh_mode)
analysis.add_system(solver.state)
analysis.add_task('La')
analysis.add_task('Lb')
analysis.add_task('Lc')
analysis.add_task('nx')
analysis.add_task('ny')
analysis.add_task('Lx')
analysis.add_task('Ly')
analysis.add_task('dim')

# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(5,5))
p = axis.pcolormesh(xm, ym, s['g'].T, cmap='RdBu_r');
axis.set_xlim([0,40.0*np.pi/np.sqrt(3.0)])
axis.set_ylim([0,24.0*np.pi])

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = initial_dt
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_array(np.ravel(s['g'][:-1,:-1].T))

        display.clear_output()
        display.display(plt.gcf())
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

p.set_array(np.ravel(s['g'][:-1,:-1].T))
display.clear_output()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

# Read in the data
f = h5py.File('analysis_tasks/analysis_tasks_s1/analysis_tasks_s1_p0.h5','r')
y = f['/scales/y/1.0'][:]
t = f['scales']['sim_time'][:]
f.close()

# Merge output
logger.info('beginning join operation')
logger.info(analysis.base_path)
post.merge_analysis(analysis.base_path)
