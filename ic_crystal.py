import numpy as np

def ic(x,y):

    eps = 0.25
    bar_field = np.sqrt(eps)/2.0
    q = np.sqrt(3.0)/2.0
    cx = 10.0*np.pi/q
    cy = 6.0*np.sqrt(3.0)*np.pi/q;
    d0 = 0.33*cx
    field = np.zeros((len(x[:,0]),len(y[0,:])))
    for i in range(len(x[:,0])):
        for j in range(len(y[0,:])):
            xl = x[i,0]; yl = y[0,j]
            field[i,j] = bar_field
            d = np.sqrt((xl-cx)**2+(yl-cy)**2)
            if (d<d0):
                w = (1.0-(d/d0)**2)**2
                A = 0.8*(bar_field+np.sqrt(15.0*eps-36.0*bar_field**2)/3.0)
                aux_field = np.cos(yl*q/np.sqrt(3.0))*np.cos(xl*q-np.pi)-0.5*np.cos(yl*2.0*q/np.sqrt(3.0))
                field[i,j] += w*A*aux_field

    return field
