#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     FTCS SCHEMES

import numpy as np
from wavefuncs import wavepacket_1d, wavepacket_2d, wavepacket_2particle
from potentials import potential_C, potential_D, potential_E
from num_aux import integrate_1d, integrate_2d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
     
    # run loop to compute FTCS scheme and write mod square of result to file
    for k in np.arange(tn):

        psi[1:xn-1] = psi[1:xn-1] + dt*1j*((psi[2:xn]-2*psi[1:xn-1]+psi[0:xn-2])/(2*dx**2) + V[1:xn-1]*psi[1:xn-1])
        psi[0] = 0
        psi[xn-1] = 0
        
        if (k in k_arr):
            T[j]   = t[k]
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
        V = np.ones((xn,yn))
    if case=="caseD":
       V = potential_D(x,y,sys_par)

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
    
    #Loop to compute FCTS scheme
    for k in np.arange(tn):

        psi = psi*V

        psi[1:xn-1,1:yn-1] = psi[1:xn-1,1:yn-1] + (dt*1j/2)*((psi[1:xn-1,2:yn]-2*psi[1:xn-1,1:yn-1]+psi[1:xn-1,0:yn-2])/(dy**2) + (psi[2:xn,1:yn-1]-2*psi[1:xn-1,1:yn-1]+psi[0:xn-2,1:yn-1])/(dx**2))


        psi[0:,0] = 0
        psi[0:, yn-1] = 0
        psi[xn-1,0:] = 0
        psi[0, 0:] = 0
         
        if (k in k_arr):
            T[j]   = t[k]
            P[j]   = np.abs(psi)**2
            val[j] = integrate_2d(P[j],x,y)
            j += 1   
                                                 
    # return output                                    
    return P, x, y, val, T

# FCTS method to solve two-particle case numerically (case E)  
def ftcs_2particle(case, settings, sys_par, num_par): 
 
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
    psi1,psi2 = wavepacket_2particle(x, sys_par)
    psi = psi1+psi2
    P = np.abs(psi)**2
    
    maxima, _ = find_peaks(P)
    
    V = potential_E(x,maxima[0],maxima[1],sys_par)
        
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

    # run loop to compute FTCS scheme and write mod square of result to file
    for k in np.arange(tn):
        psi[1:xn-1] = psi[1:xn-1] + dt*1j*((psi[2:xn]-2*psi[1:xn-1]+psi[0:xn-2])/(2*dx**2)  + V[1:xn-1]*psi[1:xn-1]) 
        psi[0] = 0
        psi[xn-1] = 0

        # update potential
        p = np.abs(psi)**2
        maxima, _ = find_peaks(p, height=0.95*np.amax(p))
        V = potential_E(x,maxima[0],maxima[1],sys_par)

        if (k in k_arr):
            T[j]   = t[k]
            P[j]   = np.abs(psi)**2
            val[j] = integrate_1d(P[j],x)
            j += 1
    
    # return output                                    
    return P, x, val, T