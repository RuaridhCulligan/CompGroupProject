#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     AUXILIARY FUNCTIONS RELATING TO BOUNDARY CONDITIONS

import numpy as np

# potential function for case C at position x
def potential_C(x, sys_params):
    
    V0 = sys_params[5]
    d  = sys_params[6]

    V = V0 * (np.abs(x) < 0.5*d)
     
    return V
    
# potential function for case D at position x,y
def potential_D(x,y, sys_params):
    
    d  = sys_params[6]
    w  = sys_params[7]

    xn = len(x)
    yn = len(y)
    
    V = np.ones((xn,yn))

    for i in range(xn):
        for j in range(yn):
            if np.abs(x[i]) < d/2 and np.abs(y[j]) > w/2:
                V[j,i] = 0

    V[0:,0] = 0
    V[0:, yn-1] = 0
    V[xn-1,0:] = 0
    V[0, 0:] = 0

    return V

# potential function for case E at position x1,x2
def potential_E(x, x1, x2, sys_params):
    
    V0    = sys_params[5]
    alpha = sys_params[8]
    
    V = np.zeros(len(x))
    V[x1:x2+1] = V0*np.exp(-alpha*np.abs(x[x1]-x[x2])**2)

    return V      