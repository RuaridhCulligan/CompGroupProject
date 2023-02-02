#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     CN SCHEMES

import numpy as np
from wavefuncs import wavepacket_1d, wavepacket_2d
from potentials import potential_C, potential_D
from num_aux import integrate_1d, integrate_2d, tridiag_solv

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning) # surpress RuntimeWarnings 

# CN method to solve one-dimensional case numerically (cases A & C)
def cn_1D(case, settings, sys_par, num_par):
     
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
    psi = wavepacket_1d(x, sys_par)
    psi.dtype = complex 
    
    # set up relevant potential at each point
    if case=="caseA":
        V = np.zeros(xn)
    if case=="caseC":
        V = potential_C(x, sys_par)
    
    # make relevant adjustments for non-static/semi-static output:
    if float(settings[0])==0:    
        T   = np.arange(t_start, t_end+dt, dt*100)
        P   = np.empty(len(T), dtype="object")
        val = np.empty(len(T), dtype="float")
        j       = 0
    if float(settings[0])==0.5: 
        T   = np.array([t_start,t_end/8,t_end/4,t_end/2, 3*t_end/4, t_end]) 
        P   = np.empty(len(T), dtype="object")
        val = np.empty(len(T))
        j   = 0
    else:
        T   = np.array([t_end])
        P   = np.empty(1, dtype="object")
        val = np.array([1], dtype="float")
        j   = 0
    #### POSSIBLY A 4 NOT 2. WAS 4 IN OTHER CODE NOT SURE WHY
    sigma = np.ones(xn)*(dt*1j)/(4*dx**2)

    A = np.diag(-sigma[0:xn-1], 1) + np.diag(1+2*sigma) + np.diag(-sigma[0:xn-1], -1)
    B = np.diag(sigma[0:xn-1], 1) + np.diag(1-2*sigma + V) + np.diag(sigma[0:xn-1], -1)

    for i in range(1,tn):
        psi = np.linalg.solve(A, B.dot(psi))
        
        psi[0] = 0; psi[xn-1] = 0

        if (t[i] in T):
            P[j]   = np.abs(psi)**2
            val[j] = integrate_1d(P[j],x)
            j += 1   
                                                 
    # return output                                    
    return P, x, val, T

# CN method to solve two-dimensional case numerically (cases B & D)
def cn_2D(case, settings, sys_par, num_par):
      
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
    psi = wavepacket_2d(x,y, sys_par)
    psi.dtype = complex 
    
    # set up relevant potential at each point
    if case=="caseB":
        V = np.zeros((xn,yn))
    if case=="caseD":
        V = potential_D(x,y,sys_par)
        
    # make relevant adjustments for non-static/semi-static output:
    if settings[2]=="0":    
        T   = np.arange(t_start, t_end+dt, dt*100)
        P   = np.empty(len(T), dtype="object")
        val = np.empty(len(T))
        j       = 0
    if settings[2]=="0.5": 
        T   = np.array([t_start,t_end/8,t_end/4,t_end/2, 3*t_end/4, t_end]) 
        P   = np.empty(len(T), dtype="object")
        val = np.empty(len(T))
        j   = 0
    else:
        T   = np.array([t_end])
        P   = np.empty((xn,yn))
        val = np.array([1])
        j   = 0

    #Loop over the time values and calculate the derivative
    sigma_y = np.ones(yn*xn - 1)*(dt*1j/(2*dy**2))
    sigma_x = np.ones(xn*(yn-1))*(dt*1j/(2*dx**2))
    sigma_xy = np.ones(yn*xn)*(dt*1j/2)*(1/dx**2 + 1/dy**2)

    A = np.diag(-sigma_y, 1) + np.diag(1+2*sigma_xy) + np.diag(-sigma_y, -1) + np.diag(-sigma_x, -yn) + np.diag(-sigma_x, yn)
    B = np.diag(sigma_y, 1) + np.diag(1-2*sigma_xy + V.flatten()) + np.diag(sigma_y, -1) + np.diag(-sigma_x, -yn) + np.diag(-sigma_x, yn)

    for i in range(1,tn):

        Psi = psi.flatten
        Psi = np.linalg.solve(A, B.dot(Psi))
        
        psi = Psi.reshape(xn,yn)
        psi[0,0:] = 0; psi[xn-1,0:] = 0; psi[0:,yn-1] = 0; psi[0:,0] = 0

        if (t[i] in T):
            P[j]   = np.abs(psi)**2
            val[j] = integrate_2d(P[j],x,y)
            j += 1   
                                                 
    # return output                                    
    return P, x, y, val, T

# CN method to solve two-particle case numerically (case E)  
def cn_2particle(case, settings, sys_par, num_par): 
    
    """
    to do!
    """
    
    return 0