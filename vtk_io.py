
import numpy as np
from tvtk.api import tvtk, write_data
import sys

def dump_scalar_to_vtk(file,nx,d,Domain,scalar_field):
    D = np.zeros(3)
    if (d==2):
        D = np.asarray([Domain[0],Domain[1],1])
        nx = np.asarray([nx[0],nx[1],1])
    if (d==3):
        D = np.asarray([Domain[0],Domain[1],Domain[2]])
        nx = np.asarray([nx[0],nx[1],nx[2]])
    # Generate some points.
    x, y, z = np.mgrid[0:D[0]:nx[0]*1j, \
                       0:D[1]:nx[1]*1j, \
                       0:D[2]:nx[2]*1j]

    # The actual points.
    pts = np.empty(x.shape+(3,), dtype=float)
    pts[...,0] = x
    pts[...,1] = y
    pts[...,2] = z

    # Scalar
    if (scalar_field.shape!=x.shape):
        print('wrong shape')

    # We reorder the points, scalars and vectors so this is as per VTK's
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size // 3, 3
    scalars = scalars.T.copy()

    # Create the dataset.
    sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
    sg.point_data.scalars = scalar_field.ravel()
    sg.point_data.scalars.name = 'scalars'

    write_data(sg, str(file)+'.vtk')
    return

def dump_vector_to_vtk(file,nx,d,Domain,fields):
    D = np.zeros(3)
    if (d==2):
        D = np.asarray([Domain[0],Domain[1],1])
        nx = np.asarray([nx[0],nx[1],1])
    if (d==3):
        D = np.asarray([Domain[0],Domain[1],Domain[2]])
        nx = np.asarray([nx[0],nx[1],nx[2]])
    # Generate some points.
    x, y, z = np.mgrid[0:D[0]:nx[0]*1j, \
                       0:D[1]:nx[1]*1j, \
                       0:D[2]:nx[2]*1j]

    # The actual points.
    pts = np.empty(x.shape+(3,), dtype=float)
    pts[...,0] = x
    pts[...,1] = y
    pts[...,2] = z

    # Vector
    if (vector_field.shape!=x.shape+(3,)):
        print('wrong shape')

    # We reorder the points, scalars and vectors so this is as per VTK's
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size // 3, 3
    vectors = vectors.transpose(2, 1, 0, 3).copy()
    vectors.shape = vectors.size // 3, 3

    # Create the dataset.
    sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
    sg.point_data.vectors = vector_field
    sg.point_data.vectors.name = 'velocity'

    write_data(sg, str(file)+'.vtk')
    return
