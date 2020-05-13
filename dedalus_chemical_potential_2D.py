#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
import time
from IPython import display
import vtk_io as vtk

from chemical_potential_2D import mu_1g

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

#Aspect ratio 2
Lx, Ly = (2., 1.)
nx, ny = (192, 96)

# Create bases and domain
dealiasx = 3/2
dealiasy = 3/2
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=dealiasx)
y_basis = de.Chebyshev('y',ny, interval=(-Ly/2, Ly/2), dealias=dealiasy)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

Schmidt = 1.e3

problem = de.IVP(domain, variables=['s','sy','mu','muy'])
problem.meta[:]['y']['dirichlet'] = True
problem.parameters['Sc'] = Schmidt
problem.add_equation("dt(s) - 1/Sc*(dx(dx(mu)) + dy(muy)) = 0")
problem.add_equation("muy - dy(mu) = 0")
problem.add_equation("sy - dy(s) = 0")
problem.add_equation(mu_1g())

problem.add_bc("left(s) = 0")
problem.add_bc("right(s) = 1")
problem.add_bc("left(mu) = 0")
problem.add_bc("right(mu) = 1")

ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
s = solver.state['s']
sy = solver.state['sy']

a = 0.05
s['g'] = 0.5*(1+np.tanh(y/a))
s.differentiate('y',out=sy)

solver.stop_sim_time = 2.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = Lx/nx

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.1, max_writes=50)
analysis.add_task('s')
solver.evaluator.vars['Lx'] = Lx

# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(10,5))
p = axis.pcolormesh(xm, ym, s['g'].T, cmap='RdBu_r');
axis.set_xlim([0,2.])
axis.set_ylim([-0.5,0.5])

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
s_ave = f['tasks']['s profile'][:]
f.close()

s_ave = s_ave[:,0,:] # remove length-one x dimension

for i in range(0,21,5):
  plt.plot(s_ave[i,:],y,label='t=%4.2f' %t[i])

plt.ylim([-0.5,0.5])
plt.xlim([0,1])
plt.xlabel(r'$\frac{\int \ s dx}{L_x}$',fontsize=24)
plt.ylabel(r'$y$',fontsize=24)
plt.legend(loc='lower right').draw_frame(False)
