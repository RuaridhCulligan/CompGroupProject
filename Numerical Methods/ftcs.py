#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     FTCS SCHEMES

import numpy as np
from wavefuncs import wavepacket_1d, wavepacket_2d
from potentials import potential_C, potential_D
from num_aux import integrate_1d, integrate_2d

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning) # surpress RuntimeWarnings 

# FCTS method to solve two-dimensional case numerically (cases A & C) 
def ftcs_1D(case, settings, sys_par, num_par):
    
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
    psi = wavepacket_1d(x, sys_par).astype("complex_")
    
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
     
    # run loop to compute FTCS scheme and write mod square of result to file
    for i in np.arange(tn):

        psi[1:xn-1] = psi[1:xn-1] + dt*1j*((psi[2:xn]-2*psi[1:xn-1]+psi[0:xn-2])/(2*dx**2) + V[1:xn-1]*psi[1:xn-1])
        psi[0] = 0
        psi[xn-1] = 0
        
        if (t[i] in T):
            P[j]   = np.abs(psi)**2
            val[j] = integrate_1d(P[j],x)
            j += 1
        
    # return output
    return P, x, val, T
    

# FCTS method to solve two-dimensional case numerically (cases B & D)  
def ftcs_2D(case, settings, sys_par, num_par):
    
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
    psi = wavepacket_2d(x,y, sys_par).astype("complex_")
    
    # set up relevant potential at each point
    if case=="caseB":
        V = np.zeros((xn,yn))
    if case=="caseD":
        V = potential_D(x,y, sys_par)
        
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
        P   = np.empty(len(T), dtype="object")
        val = np.array([1])
        j   = 0    
    
    #Loop to compute FCTS scheme
    for k in np.arange(tn):
        for i in np.arange(xn-1):
            
            psi[1:xn-1,1:yn-1] = psi[1:xn-1,1:yn-1] + (dt*1j/2)*((psi[1:xn-1,2:yn]-2*psi[1:xn-1,1:yn-1]+psi[1:xn,0:yn-2])/(dy**2) + (psi[2:xn,1:yn-1]-2*psi[1:xn-1,1:yn-1]+psi[0:xn-2,1:yn-1])/(dx**2) + V[1:xn-1,1:yn-1]*psi[1:xn-1,1:yn-1])
            psi[:,0] = 0
            psi[:, yn-1] = 0
            psi[xn-1,:] = 0
            psi[0, :] = 0
            
        if (t[k] in T):
            P[j]   = np.abs(psi)**2
            val[j] = integrate_2d(P[j],x,y)
            j += 1   
                                                 
    # return output                                    
    return P, x, y, val, T

# FCTS method to solve two-particle case numerically (case E)  
def ftcs_2particle(case, settings, sys_par, num_par): 
    
    """
    to do!
    """
    
    return 0