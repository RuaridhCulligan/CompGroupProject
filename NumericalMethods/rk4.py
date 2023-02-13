#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     RK4 SCHEMES

import numpy as np
from NumericalMethods.wavefuncs import wavepacket_1d, wavepacket_2d
from NumericalMethods.potentials import potential_C, potential_D
from NumericalMethods.num_aux import integrate_1d, integrate_2d

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning) # surpress RuntimeWarnings 

# RK4 method to solve one-dimensional case numerically (cases A & C)
def rk4_1D(case, settings, sys_par, num_par):
     
    # set up grids
    x_min = num_par[0]
    x_max = num_par[1]
    dx    = num_par[2]
    
    t_start = 0
    t_end   = sys_par[0]
    dt      = num_par[3]
    
    t = np.arange(t_start, t_end+dt, dt)
    x = np.arange(x_min, x_max+dx, dx)
    
    # determine grid lengths
    tn = len(t)
    xn = len(x)
    
    # initialise wavefunction 
    psi = wavepacket_1d(x, sys_par).astype("complex")
    
    # set up relevant potential at each point
    if case=="caseA":
        V = np.zeros(xn)
    if case=="caseC":
        V = potential_C(x, sys_par)
    
    # make relevant adjustments for non-static/semi-static output:
    if float(settings[0]) == 0.0:  
        k_arr = np.linspace(0, tn-1, 100, dtype="int")  
        T     = np.empty(len(k_arr))
        P     = np.empty(len(T), dtype="object")
        val   = np.empty(len(T))
        j     = 0
    if float(settings[0]) == 0.5: 
        k_arr = np.array([0,(tn-1)/8, (tn-1)/4 ,(tn-1)/2,3*(tn -1)/4 ,tn-1], dtype="int")
        T     = np.empty(len(k_arr)) 
        P     = np.empty(len(T), dtype="object")
        val   = np.empty(len(T))
        j     = 0
    if float(settings[0]) == 1.0:
        k_arr = np.array([tn-1])
        T     = np.array([t_end])
        P     = np.empty(len(T), dtype="object")
        val   = np.array([1])
        j     = 0 
     
    #Loop over the time values and calculate the derivative
    for k in range(1,tn):
        f = np.zeros(xn).astype("complex"); k1 = np.zeros(xn).astype("complex");  k2 = np.zeros(xn).astype("complex"); k3 = np.zeros(xn).astype("complex"); k4 = np.zeros(xn).astype("complex")

        f[1:xn-1] = (1j/2)*(psi[0:xn-2]-2*psi[1:xn-1]+psi[2:xn])/(dx**2) + V[1:xn-1]*psi[1:xn-1]

        k1 = f

        k2[1:xn-1] = f[1:xn-1] + (1j/4)*(k1[0:xn-2]-2*k1[1:xn-1]+k1[2:xn])/(dx**2) + V[1:xn-1]*(psi[1:xn-1] + k1[1:xn-1])

        k3[1:xn-1] = f[1:xn-1] + (1j/4)*(k2[0:xn-2]-2*k2[1:xn-1]+k2[2:xn])/(dx**2) + V[1:xn-1]*(psi[1:xn-1] + k2[1:xn-1])

        k4[1:xn-1] = f[1:xn-1] + (1j/2)*(k3[0:xn-2]-2*k3[1:xn-1]+k3[2:xn])/(dx**2) + V[1:xn-1]*(psi[1:xn-1] + k2[1:xn-1])

        psi = psi + dt*(k1/6 + k2/3 + k3/3 + k4/6)

        #Force boundary conditions on the off chance something has gone wrong and they contain a value
        psi[0] = 0; psi[xn-1] = 0

        if (k in k_arr):
            T[j]   = t[k]
            P[j]   = np.abs(psi)**2
            val[j] = integrate_1d(P[j],x)
            j += 1

    
    return P, x, val, T

# RK4 method to solve two-dimensional case numerically (cases B & D)
def rk4_2D(case, settings, sys_par, num_par):
     
    # set up grids
    x_min = num_par[0]
    x_max = num_par[1]
    dx    = num_par[2]
    
    y_min = num_par[4]
    y_max = num_par[5]
    dy    = num_par[6]
    
    t_start = 0
    t_end   = sys_par[0]
    dt      = num_par[3]
    
    t = np.arange(t_start, t_end+dt, dt)
    x = np.arange(x_min, x_max+dx, dx)
    y = np.arange(y_min, y_max+dy, dy)
    
    # determine grid lengths
    tn = len(t)
    xn = len(x)
    yn = len(y)
    
    # initialise wavefunction 
    psi = wavepacket_2d(x,y, sys_par).astype("complex") 
    
    # set up relevant potential at each point
    if case=="caseB":
        V = np.zeros(xn)
    if case=="caseD":
        V = potential_D(x,sys_par)
        
    # make relevant adjustments for non-static/semi-static output:
    if float(settings[0]) == 0.0:  
        k_arr = np.linspace(0, tn-1, 100, dtype="int")  
        T     = np.empty(len(k_arr))
        P     = np.empty(len(T), dtype="object")
        val   = np.empty(len(T))
        j     = 0
    if float(settings[0]) == 0.5: 
        k_arr = np.array([0,(tn-1)/8 ,(tn-1)/4 ,(tn-1)/2,3*(tn -1)/4 ,tn-1], dtype="int")
        T     = np.empty(len(k_arr)) 
        P     = np.empty(len(T), dtype="object")
        val   = np.empty(len(T))
        j     = 0
    if float(settings[0]) == 1.0:
        k_arr = np.array([tn-1])
        T     = np.array([t_end])
        P     = np.empty(len(T), dtype="object")
        val   = np.array([1])
        j     = 0

    #Loop over the time values and calculate the derivative
    for k in range(1,tn):
        for i in range(1,xn):
            f = np.zeros((xn,yn)).astype("complex"); k1 = np.zeros((xn,yn)).astype("complex");  k2 = np.zeros((xn,yn)).astype("complex"); k3 = np.zeros((xn,yn)).astype("complex"); k4 = np.zeros((xn,yn)).astype("complex")

            f[1:xn-1] = (1j/2)*(phi[0:xn-2]-2*phi[1:xn-1]+phi[2:xn])/(dx**2)# + V[1:xn-1]*phi[1:xn-1]

            k1 = f

            k2[1:xn-1] = f[1:xn-1] + (1j/4)*(k1[0:xn-2]-2*k1[1:xn-1]+k1[2:xn])/(dx**2)# + V[1:xn-1]*(phi[1:xn-1] + k1[1:xn-1])

            k3[1:xn-1] = f[1:xn-1] + (1j/4)*(k2[0:xn-2]-2*k2[1:xn-1]+k2[2:xn])/(dx**2)# + V[1:xn-1]*(phi[1:xn-1] + k2[1:xn-1])

            k4[1:xn-1] = f[1:xn-1] + (1j/2)*(k3[0:xn-2]-2*k3[1:xn-1]+k3[2:xn])/(dx**2)# + V[1:xn-1]*(phi[1:xn-1] + k2[1:xn-1])

            phi = phi + dt*(k1/6 + k2/3 + k3/3 + k4/6)

        #Force boundary conditions on the off chance something has gone wrong and they contain a value
        phi[0] = 0; phi[xn-1] = 0

        if (k in k_arr):
            T[j]   = t[k]
            P[j]   = np.abs(psi)**2
            val[j] = integrate_1d(P[j],x)
            j += 1   
                                                 
    # return output                                    
    return P, x, y, val, T

# RK4 method to solve two-particle case numerically (case E)  
def rk4_2particle(case, settings, sys_par, num_par): 
    
    """
    to do!
    """
    
    return 0