#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
import time
from IPython import display
import vtk_io as vtk

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

#Aspect ratio 2
Lx, Ly = (2., 1.)
nx, ny = (192, 96)
dim = 2

# Create bases and domain
dealiasx = 3/2
dealiasy = 3/2
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=dealiasx)
y_basis = de.Chebyshev('y',ny, interval=(-Ly/2, Ly/2), dealias=dealiasy)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

Reynolds = 1e4
Schmidt = 1.

problem = de.IVP(domain, variables=['p','u','v','uy','vy','s','sy'])
problem.meta[:]['y']['dirichlet'] = True
problem.parameters['Re']  = Reynolds
problem.parameters['Sc']  = Schmidt
problem.parameters['nx']  = nx
problem.parameters['ny']  = ny
problem.parameters['Lx']  = Lx
problem.parameters['Ly']  = Ly
problem.parameters['dim'] = dim
problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(s) - 1/(Re*Sc)*(dx(dx(s)) + dy(sy)) = - u*dx(s) - v*sy")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("sy - dy(s) = 0")

problem.add_bc("left(u) = 0.5")
problem.add_bc("right(u) = -0.5")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0)")
problem.add_bc("left(s) = 0")
problem.add_bc("right(s) = 1")

ts = de.timesteppers.RK443

solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
s = solver.state['s']
sy = solver.state['sy']

a = 0.05
sigma = 0.2
flow = -0.5
amp = -0.2
u['g'] = flow*np.tanh(y/a)
v['g'] = amp*np.sin(2.0*np.pi*x/Lx)*np.exp(-(y*y)/(sigma*sigma))
s['g'] = 0.5*(1+np.tanh(y/a))
u.differentiate('y',out=uy)
v.differentiate('y',out=vy)
s.differentiate('y',out=sy)

solver.stop_sim_time = 2.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks',sim_dt=0.1, max_writes=50)
analysis.add_task('s')
analysis.add_task('u')
analysis.add_task('v')
analysis.add_task('Re')
analysis.add_task('Sc')
analysis.add_task('nx')
analysis.add_task('ny')
analysis.add_task('Lx')
analysis.add_task('Ly')
analysis.add_task('dim')

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
    dt = cfl.compute_dt()
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

# Merge output
logger.info('beginning join operation')
logger.info(analysis.base_path)
post.merge_analysis(analysis.base_path)
