from sympy import Symbol, Function
from sympy.calculus.euler import euler_equations
from sympy import tensorcontraction, tensorproduct, eye
from sympy.tensor.array import derive_by_array
from sympy import init_printing
init_printing()

#---------------------––––––---------------------------------------------------#

def mu_1g(La,Lb):
    t     = Symbol('t')
    x     = Symbol('x')
    y     = Symbol('y')

    phi   = Function('phi')(x,y,t)
    gradient_1_phi = Function('gradient_1_phi')(x,y,t)

    grad_1_phi = derive_by_array(phi,[x,y])
    grad_1_phi
    interface = Lb*0.5*tensorcontraction(tensorproduct(grad_1_phi,grad_1_phi),(0, 1))
    potential_CH = La*(phi**2)*(1-phi)**2
    potential_CH
    regularizers_CH = euler_equations(interface,[phi,gradient_1_phi],[x,y,t])
    regularizers_CH
    mu_CH_b = str(- potential_CH.diff(phi))
    mu_CH_i = str(regularizers_CH[0])

    old = []
    old.append("Derivative(phi(x, y, t), (x, 2))")
    old.append("Derivative(phi(x, y, t), (y, 2))")
    old.append("phi(x, y, t)")

    new = []
    new.append("dx(dx(s))")
    new.append("dy(sy)")
    new.append("s")
    add_eq = 'dy(s) - sy = 0'

    for i in range(len(old)):
        mu_CH_b = mu_CH_b.replace(old[i],new[i])
        mu_CH_i = mu_CH_i.replace(old[i],new[i])
    mu_CH_i = mu_CH_i.replace("Eq(","")
    mu_CH = mu_CH_i.replace(", 0)"," - mu = ")+mu_CH_b
    return mu_CH

#---------------------––––––---------------------------------------------------#

def mu_2g(La,Lb,Lc):
    t     = Symbol('t')
    x     = Symbol('x')
    y     = Symbol('y')

    phi   = Function('phi')(x,y,z,t)
    gradient_1_phi = Function('gradient_1_phi')(x,y,t)
    gradient_2_phi = Function('gradient_2_phi')(x,y,t)

    grad_1_phi = derive_by_array(phi,[x,y])
    grad_2_phi = derive_by_array(grad_1_phi,[x,y])
    grad_2_phi
    interface = Lb*tensorcontraction(tensorproduct(grad_1_phi,grad_1_phi), (0, 1))\
        -0.5*Lc*tensorcontraction(tensorproduct(grad_2_phi,grad_2_phi),(0,1,2,3))
    potential_PFC = La*(phi**2)*(1-phi)**2+interface
    regularizers_PFC = euler_equations(potential_PFC, [phi,gradient_1_phi,gradient_2_phi], [x,y,t])
    regularizers_PFC
    mu_PFC_b = str(- potential_PFC.diff(phi))
    mu_PFC_i = str(regularizers_PFC[0])

    old = []
    old.append("Derivative(phi(x, y, t), (x, 2))")
    old.append("Derivative(phi(x, y, t), (y, 2))")
    old.append("Derivative(phi(x, y, t), (x, 4))")
    old.append("Derivative(phi(x, y, t), (y, 4))")
    old.append("phi(x, y, t)")

    new = []
    new.append("dx(dx(s))")
    new.append("dy(sy)")
    new.append("dx(dx(dx(dx(s))))")
    new.append("dy(dy(dy(sy)))")
    new.append("s")
    add_eq = 'dy(s) - sy = 0'

    for i in range(len(old)):
        mu_PFC_b = mu_PFC_b.replace(old[i],new[i])
        mu_PFC_i = mu_PFC_i.replace(old[i],new[i])
    mu_PFC_i = mu_PFC_i.replace("Eq(","")
    mu_PFC = mu_PFC_i.replace(", 0)"," - mu = ")+mu_PFC_b
    return mu_PFC
