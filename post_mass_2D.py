import numpy as np
import h5py
import vtk_io as vtk

#['scales']['Ty', 'constant', 'iteration', 'kx', 'sim_time', 'timestep', 'wall_time', 'world_time', 'write_number', 'x', 'y']

#vtk_io
f = h5py.File('analysis_tasks/analysis_tasks_s1/analysis_tasks_s1_p0.h5','r')
ts_ = f['scales']['timestep']
dim = int(f['tasks']['dim'][0])
dom = [f['tasks']['Lx'][0],f['tasks']['Ly'][0]]
nx  = [int(f['tasks']['nx'][0]),int(f['tasks']['ny'][0])]
for i in range(len(list(ts_))):
    file = '-mass-'+str(i)
    scalar = np.asarray(f['tasks']['s'][i])
    vtk.dump_scalar_to_vtk('s'+file,nx,dim,dom,scalar)
