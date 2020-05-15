import numpy as np
import h5py
import vtk_io as vtk

#['scales']['Ty', 'constant', 'iteration', 'kx', 'sim_time', 'timestep', 'wall_time', 'world_time', 'write_number', 'x', 'y']

#vtk_io
f = h5py.File('analysis_tasks/analysis_tasks_s1.h5','r')
ts_ = f['scales']['timestep']
dim = int(f['tasks']['dim'][0])
dom = [f['tasks']['Lx'][0],f['tasks']['Ly'][0]]
nx  = [int(f['tasks']['nx'][0]),int(f['tasks']['ny'][0])]
for i in range(len(list(ts_))):
    file = '-KH-'+str(i)
    u_1 = np.asarray(f['tasks']['u'][i])
    print (u_1.shape)
    u_2 = np.asarray(f['tasks']['v'][i])
    u_3 = np.zeros((nx[0],nx[1]))
    velocity = np.asarray([u_1,u_2,u_3]).transpose(1, 2, 0)
    vtk.dump_vector_to_vtk('u'+file,nx,dim,dom,velocity)
    scalar   = np.asarray(f['tasks']['s'][i])
    vtk.dump_scalar_to_vtk('s'+file,nx,dim,dom,scalar)