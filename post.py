import numpy as np
import h5py
import vtk_io as vtk

#Aspect ratio 2
d = 2
Lx, Ly = (2., 1.)
nx  = [192, 96]
dom = [Lx,Ly]

#vtk_io
f = h5py.File('analysis_tasks/analysis_tasks_s1/analysis_tasks_s1_p0.h5','r')
ts_ = f['scales']['timestep']
for i in range(len(list(ts_))):
    file = '-KH-'+str(i)
    u_1 = np.asarray(f['tasks']['u'][i]).reshape(nx[0],nx[1],1)
    u_2 = np.asarray(f['tasks']['v'][i]).reshape(nx[0],nx[1],1)
    u_3 = np.zeros((nx[0],nx[1],1))
    velocity = np.concatenate((u_1,u_2,u_3)).reshape(nx[0],nx[1],1,3)
    vtk.dump_vector_to_vtk('u'+file,nx,d,dom,velocity)
    scalar   = np.asarray(f['tasks']['s'][i])
    vtk.dump_scalar_to_vtk('s'+file,nx,d,dom,scalar)
    #scalar   = np.asarray(f['tasks']['u'][i])
    #vtk.dump_scalar_to_vtk('u'+file,nx,d,dom,scalar)
    #scalar   = np.asarray(f['tasks']['v'][i])
    #vtk.dump_scalar_to_vtk('v'+file,nx,d,dom,scalar)
