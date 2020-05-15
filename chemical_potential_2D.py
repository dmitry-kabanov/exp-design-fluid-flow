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
    regularizers_CH = Lb*0.5*tensorcontraction(tensorproduct(grad_1_phi,grad_1_phi),(0, 1))
    potential_CH = La*(phi**2)*(1-phi)**2
    d_regularizers_CH = euler_equations(regularizers_CH,[phi,gradient_1_phi],[x,y,t])
    mu_CH_rhs = str(- potential_CH.diff(phi))
    mu_CH_lhs = str(d_regularizers_CH[0])

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
        mu_CH_rhs = mu_CH_rhs.replace(old[i],new[i])
        mu_CH_lhs = mu_CH_lhs.replace(old[i],new[i])
    mu_CH_lhs = mu_CH_lhs.replace("Eq(","")
    mu_CH = mu_CH_lhs.replace(", 0)"," - mu = ")+mu_CH_rhs
    return mu_CH

#---------------------––––––---------------------------------------------------#

def mu_2g(La,Lb,Lc):
    t     = Symbol('t')
    x     = Symbol('x')
    y     = Symbol('y')

    phi   = Function('phi')(x,y,t)
    gradient_1_phi = Function('gradient_1_phi')(x,y,t)
    gradient_2_phi = Function('gradient_2_phi')(x,y,t)

    grad_1_phi = derive_by_array(phi,[x,y])
    grad_2_phi = derive_by_array(grad_1_phi,[x,y])
    grad_2_phi
    regularizers_PFC_1 = Lb*tensorcontraction(tensorproduct(grad_1_phi,grad_1_phi), (0, 1))
    regularizers_PFC_2 = -0.5*Lc*tensorcontraction(tensorproduct(grad_2_phi,grad_2_phi),(0,1,2,3))
    potential_PFC = La*(phi**2)*(1-phi)**2
    d_regularizers_PFC_1 = euler_equations(regularizers_PFC_1, [phi,gradient_1_phi,gradient_2_phi], [x,y,t])
    d_regularizers_PFC_2 = euler_equations(regularizers_PFC_2, [phi,gradient_1_phi,gradient_2_phi], [x,y,t])

    d_regularizers_PFC_2 = str(d_regularizers_PFC_2[0]).replace("Eq(","")
    d_regularizers_PFC_2 = d_regularizers_PFC_2.replace(", 0)","")
    mu_PFC_rhs = str(- potential_PFC.diff(phi))+d_regularizers_PFC_2
    mu_PFC_lhs = str(d_regularizers_PFC_1[0])

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
        mu_PFC_rhs = mu_PFC_rhs.replace(old[i],new[i])
        mu_PFC_lhs = mu_PFC_lhs.replace(old[i],new[i])
    mu_PFC_lhs = mu_PFC_lhs.replace("Eq(","")
    mu_PFC = mu_PFC_lhs.replace(", 0)"," - mu = ")+mu_PFC_rhs
    return mu_PFC
