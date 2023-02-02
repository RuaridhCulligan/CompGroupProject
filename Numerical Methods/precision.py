#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     FUNCTIONS TO EVALUATE THE PRECISION OF NUMERICAL METHODS IN CASES A AND B

# import modules
import numpy as np
import matplotlib.pyplot as plt
import time 
import psutil

# import functions
from log_handling import read_log
from ftcs import ftcs_1D, ftcs_2D, ftcs_2particle
from rk4 import rk4_1D, rk4_2D, rk4_2particle 
from cn import cn_1D, cn_2D, cn_2particle
from wavefuncs import an_sol_1D, an_sol_2D
from num_aux import integrate_1d, integrate_2d

# set standardised layout of plots
fig_dim    = [16, 8]   # dimensions
title_size = 16       # title font size
body_size  = 14       # axes and legends font size
tick_size  = 12       # tick mark font size 
plt.rcParams['text.usetex'] = True # enable LaTeX renadering
plt.rcParams['mathtext.fontset'] = 'cm' # use LateX font for maths
plt.rcParams['font.family'] = 'STIXGeneral' # use LateX font for text

# define "standard set-up" for comparison of methods
t_end = 1
a     = 1 
kx0    = 1 
b     = 1 
ky0   = 1 
x0    = 0 
y0    = 0 
x_min = -50 
x_max = +50 
y_min = -10 
y_max = +10 

# given values for dx, dt, dy,, case package into relevant arrays
def package(case, dx, dt, dy):
    
    # set up arrays
    num_par = np.ones(7)
    sys_params = np.ones(11)
    settings = np.ones(6)

    # fill arrays
    num_par[0] = x_min
    num_par[1] = x_max
    num_par[2] = dx
    num_par[3] = dt
    num_par[6] = dy
    num_par[4] = y_min
    num_par[5] = y_max

    sys_params[1] = a
    sys_params[2] = kx0
    sys_params[3] = b 
    sys_params[4] = ky0 
    sys_params[9] = x0 
    sys_params[10] = y0

    # return in same format as "read_log":
    return case, settings, sys_params, num_par


# for a given case, method, dt, dx, dy find total error
def total_error(case, method, dt, dx, dy):

    case,  settings, sys_params, num_par = package(case, dx, dt, dy)

    if case=="caseA":
        if method=="ftcs":
            start = time.time() 
            P, x, _, _ = ftcs_1D(case, settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x, _, _ = rk4_1D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="cn":
            start = time.time()
            P, x, _, _ = cn_1D(case,  settings, sys_params, num_par) 
            stop = time.time()       

        P_an = np.abs(an_sol_1D(x,t_end, sys_params))**2

    if case=="caseB":
        if method=="ftcs":
            start = time.time()
            P, x, y, _, _ = ftcs_2D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x, y, _, _ = rk4_2D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="cn":
            start = time.time()
            P, x, y, _, _ = cn_2D(case,  settings, sys_params, num_par) 
            stop = time.time()     

        P_an =np.abs(an_sol_2D(x,y,t_end, case, method, settings, sys_params, num_par))**2  

    P_diff = np.abs( P_an - P[0] )

    if case=="caseA":
        err = integrate_1d(P_diff, x)

    if case=="caseB":
        err = integrate_2d(P_diff, x, y)

    return err, stop-start



case="caseA"
dx = 0.0001
dy = dx
dt = 0.0001
method = "ftcs"


err, T =total_error(case, method, dt, dx, dy)

print(err, T)

