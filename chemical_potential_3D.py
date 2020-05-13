from sympy import Symbol, Function
from sympy.calculus.euler import euler_equations
from sympy import tensorcontraction, tensorproduct, eye
from sympy.tensor.array import derive_by_array
from sympy import init_printing
init_printing()

#---------------------––––––---------------------------------------------------#

def mu_1g():
    t     = Symbol('t')
    x     = Symbol('x')
    y     = Symbol('y')
    z     = Symbol('z')

    phi   = Function('phi')(x,y,z,t)
    gradient_1_phi = Function('gradient_1_phi')(x,y,z,t)

    grad_1_phi = derive_by_array(phi,[x,y,z])
    grad_1_phi
    interface = 0.5*tensorcontraction(tensorproduct(grad_1_phi,grad_1_phi),(0, 1))
    potential_CH = (phi**2)*(1-phi)**2+interface
    potential_CH
    chemical_potential_CH = euler_equations(potential_CH,[phi,gradient_1_phi],[x,y,z,t])
    chemical_potential_CH
    mu_CH = str(chemical_potential_CH[0])

    old = []
    old.append("Derivative(phi(x, y, z, t), (x, 2))")
    old.append("Derivative(phi(x, y, z, t), (y, 2))")
    old.append("Derivative(phi(x, y, z, t), (z, 2))")
    old.append("phi(x, y, z, t)")

    new = []
    new.append("dx(dx(s))")
    new.append("dy(dy(s))")
    new.append("dz(sz)")
    new.append("s")
    add_eq = 'dz(s) - sz = 0'

    for i in range(len(old)):
        mu_CH = mu_CH.replace(old[i],new[i])
    mu_CH = mu_CH.replace("Eq(","mu = ")
    mu_CH = mu_CH.replace(", 0)","")
    return mu_CH

#---------------------––––––---------------------------------------------------#

def mu_2g():
    t     = Symbol('t')
    x     = Symbol('x')
    y     = Symbol('y')
    z     = Symbol('z')

    phi   = Function('phi')(x,y,z,t)
    gradient_1_phi = Function('gradient_1_phi')(x,y,z,t)
    gradient_2_phi = Function('gradient_2_phi')(x,y,z,t)

    grad_1_phi = derive_by_array(phi,[x,y,z])
    grad_2_phi = derive_by_array(grad_1_phi,[x,y,z])
    grad_2_phi
    interface = tensorcontraction(tensorproduct(grad_1_phi,grad_1_phi), (0, 1))\
        -0.5*tensorcontraction(tensorproduct(grad_2_phi,grad_2_phi),(0,1,2,3))
    potential_PFC = (phi**2)*(1-phi)**2+interface
    chemical_potential_PFC = euler_equations(potential_PFC, [phi,gradient_1_phi,gradient_2_phi], [x,y,z,t])
    chemical_potential_PFC
    mu_PFC = str(chemical_potential_PFC[0])

    old = []
    old.append("Derivative(phi(x, y, z, t), (x, 2))")
    old.append("Derivative(phi(x, y, z, t), (y, 2))")
    old.append("Derivative(phi(x, y, z, t), (z, 2))")
    old.append("Derivative(phi(x, y, z, t), (x, 4))")
    old.append("Derivative(phi(x, y, z, t), (y, 4))")
    old.append("Derivative(phi(x, y, z, t), (z, 4))")
    old.append("phi(x, y, z, t)")

    new = []
    new.append("dx(dx(s))")
    new.append("dy(dy(s))")
    new.append("dz(sz)")
    new.append("dx(dx(dx(dx(s))))")
    new.append("dy(dy(dy(dy(s))))")
    new.append("dz(dz(dz(sz)))")
    new.append("s")
    add_eq = 'dz(s) - sz = 0'

    for i in range(len(old)):
        mu_PFC = mu_PFC.replace(old[i],new[i])
    mu_PFC = mu_PFC.replace("Eq(","mu = ")
    mu_PFC = mu_PFC.replace(", 0)","")
    return mu_PFC
