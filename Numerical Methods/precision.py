#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     FUNCTIONS TO EVALUATE THE PRECISION OF NUMERICAL METHODS IN CASES A AND B

# import modules
import numpy as np
import matplotlib.pyplot as plt
import time 
import psutil
import os

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

# set output path
out_dir   = "output"

if os.path.exists(out_dir) ==False:
    os.mkdir(out_dir) 

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

    cpu_frac = psutil.cpu_percent(0.5)

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

    return err, stop-start, cpu_frac

# for constant time step, loop through array of spatial step sizes (assuming dx=dy)
def space_loop(case, method, dt, dxdy_arr):

    err_arr     = np.empty(len(dxdy_arr))
    runtime_arr = np.empty(len(dxdy_arr))
    cpu_arr     = np.empty(len(dxdy_arr))

    for i in np.arange(len(dxdy_arr)):
        dx, dy = dxdy_arr[i], dxdy_arr[i]
        err_arr[i], runtime_arr[i], cpu_arr[i] = total_error(case, method, dt, dx, dy)

    return err_arr, runtime_arr, cpu_arr   

# for constant space step, loop through array of time step sizes (assuming dx=dy)
def time_loop(case, method, dxdy, dt_arr):

    err_arr     = np.empty(len(dt_arr))
    runtime_arr = np.empty(len(dt_arr))
    cpu_arr     = np.empty(len(dt_arr))

    for i in np.arange(len(dt_arr)):
        dt = dt_arr[i]
        dx, dy = dxdy, dxdy
        err_arr[i], runtime_arr[i], cpu_arr[i] = total_error(case, method, dt, dx, dy)

    return err_arr, runtime_arr, cpu_arr       

# loop through space and time with "central step" dxdy0 or dt0
def double_loop(case, method,dxy0, dt0, dxdy_arr,dt_arr):  

    space_err, space_runtime, space_cpu = space_loop(case, method, dt0, dxdy_arr)
    time_err, time_runtime, time_cpu =time_loop(case, method, dxy0, dt_arr)

    arr = np.empty((2,3), dtype="object")
    arr[0,0] = space_err 
    arr[0,1] = space_runtime  
    arr[0,2] = space_cpu
    arr[1,0] = time_err 
    arr[1,1] = time_runtime  
    arr[1,2] = time_cpu

    return arr 

def run(case, method, dxdy0, dt0, dxdy_arr, dt_arr):

    if method=="all":
        arr_ftcs = double_loop(case, "ftcs", dxdy0, dt0, dxdy_arr, dt_arr)
        arr_rk4 = double_loop(case, "rk4", dxdy0, dt0, dxdy_arr, dt_arr)
        arr_cn = double_loop(case, "cn", dxdy0, dt0, dxdy_arr, dt_arr)

        return arr_ftcs, arr_rk4, arr_cn

    elif method=="ftcs" or method=="rk4" or method=="cn":
        arr = double_loop(case, method, dxdy0, dt0, dxdy_arr, dt_arr)
   
        return arr

# visualise caseA
def vis_caseA(method, dx0, dt0, dx_arr, dt_arr):

    # get data
    if method=="all":
        arr_ftcs, arr_rk4, arr_cn = run("caseA", method, dx0, dt0, dx_arr, dt_arr)
    else: 
        arr = run("caseA", method, dx0, dt0, dx_arr, dt_arr)

    # set up figure
    fig, axs = plt.subplots(3,2,figsize = [18,10])
    fig.subplots_adjust(hspace = .5, wspace=.4) 
    axs = axs.ravel()  

    plt.suptitle(r'Evaluation of Numerical Schemes in 1D', fontsize=title_size)
    
    # plot err vs dx at constant dt:
    axs[0].set_title(r'Total error against spatial step size with $dt={0:.1e}$'.format(dt0), fontsize=body_size)
    
    if method=="ftcs":
        axs[0].scatter(dx_arr, arr[0,0] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[0].scatter(dx_arr, arr[0,0] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[0].scatter(dx_arr, arr[0,0] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[0].scatter(dx_arr, arr_ftcs[0,0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[0].scatter(dx_arr, arr_rk4[0,0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[0].scatter(dx_arr, arr_cn[0,0] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[0].legend(fontsize=body_size, loc="upper center")
    axs[0].set_ylabel(r'$\epsilon_{tot}$', fontsize=body_size)
    axs[0].set_xlabel(r'$dx$', fontsize=body_size)

    # plot comp time vs dx at constant dt:
    axs[2].set_title(r'Computational time against spatial step size with $dt={0:.1e}$'.format(dt0), fontsize=body_size)
    
    if method=="ftcs":
        axs[2].scatter(dx_arr, arr[0,1] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[2].scatter(dx_arr, arr[0,1] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[2].scatter(dx_arr, arr[0,1] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[2].scatter(dx_arr, arr_ftcs[0,1] , label=r'FTCS scheme', color="black",  marker=".")
        axs[2].scatter(dx_arr, arr_rk4[0,1] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[2].scatter(dx_arr, arr_cn[0,1] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[2].legend(fontsize=body_size, loc="upper center")
    axs[2].set_ylabel(r'$T$ (s)', fontsize=body_size)
    axs[2].set_xlabel(r'$dx$', fontsize=body_size)

    # plot cpu usage vs dx at constant dt:
    axs[4].set_title(r'CPU usage against spatial step size with $dt={0:.1e}$'.format(dt0), fontsize=body_size)
    
    if method=="ftcs":
        axs[4].scatter(dx_arr, arr[0,2] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[4].scatter(dx_arr, arr[0,2] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[4].scatter(dx_arr, arr[0,2] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[4].scatter(dx_arr, arr_ftcs[0,2] , label=r'FTCS scheme', color="black",  marker=".")
        axs[4].scatter(dx_arr, arr_rk4[0,2] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[4].scatter(dx_arr, arr_cn[0,2] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[4].legend(fontsize=body_size, loc="upper center")
    axs[4].set_ylabel(r'\% CPU ', fontsize=body_size)
    axs[4].set_xlabel(r'$dx$', fontsize=body_size)

    # plot err vs dt at constant dx:
    axs[1].set_title(r'Total error against temporal step size with $dx={0:.1e}$'.format(dx0), fontsize=body_size)
    
    if method=="ftcs":
        axs[1].scatter(dt_arr, arr[1,0] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[1].scatter(dt_arr, arr[1,0] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[1].scatter(dt_arr, arr[1,0] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[1].scatter(dt_arr, arr_ftcs[1,0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[1].scatter(dt_arr, arr_rk4[1,0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[1].scatter(dt_arr, arr_cn[1,0] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[1].legend(fontsize=body_size, loc="upper center")
    axs[1].set_ylabel(r'$\epsilon_{tot}$', fontsize=body_size)
    axs[1].set_xlabel(r'$dt$', fontsize=body_size)

    # plot comp time vs dx at constant dt:
    axs[3].set_title(r'Computational time against temporal step size with $dx={0:.1e}$'.format(dx0), fontsize=body_size)
    
    if method=="ftcs":
        axs[3].scatter(dt_arr, arr[1,1] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[3].scatter(dt_arr, arr[1,1] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[3].scatter(dt_arr, arr[1,1] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[3].scatter(dt_arr, arr_ftcs[1,1] , label=r'FTCS scheme', color="black",  marker=".")
        axs[3].scatter(dt_arr, arr_rk4[1,1] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[3].scatter(dt_arr, arr_cn[1,1] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[3].legend(fontsize=body_size, loc="upper center")
    axs[3].set_ylabel(r'$T$ (s)', fontsize=body_size)
    axs[3].set_xlabel(r'$dt$', fontsize=body_size)

    # plot cpu usage vs dx at constant dt:
    axs[5].set_title(r'CPU usage against temporal step size with $dx={0:.1e}$'.format(dx0), fontsize=body_size)
    
    if method=="ftcs":
        axs[5].scatter(dt_arr, arr[1,2] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[5].scatter(dt_arr, arr[1,2] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[5].scatter(dt_arr, arr[1,2] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[5].scatter(dt_arr, arr_ftcs[1,2] , label=r'FTCS scheme', color="black",  marker=".")
        axs[5].scatter(dt_arr, arr_rk4[1,2] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[5].scatter(dt_arr, arr_cn[1,2] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[5].legend(fontsize=body_size, loc="upper center")
    axs[5].set_ylabel(r'\% CPU ', fontsize=body_size)
    axs[5].set_xlabel(r'$dt$', fontsize=body_size)

    fig.savefig(os.path.join(out_dir, "precision1D.pdf"))
    fig.show()

# visualise caseB
def vis_caseB(method, dx0, dt0, dx_arr, dt_arr):

    # get data
    if method=="all":
        arr_ftcs, arr_rk4, arr_cn = run("caseB", method, dx0, dt0, dx_arr, dt_arr)
    else: 
        arr = run("caseB", method, dx0, dt0, dx_arr, dt_arr)

    # set up figure
    fig, axs = plt.subplots(3,2,figsize = [18,10])
    fig.subplots_adjust(hspace = .5, wspace=.4) 
    axs = axs.ravel()  

    plt.suptitle(r'Evaluation of Numerical Schemes in 1D', fontsize=title_size)
    
    # plot err vs dx at constant dt:
    axs[0].set_title(r'Total error against spatial step size with $dt={0:.1e}$'.format(dt0), fontsize=body_size)
    
    if method=="ftcs":
        axs[0].scatter(dx_arr, arr[0,0] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[0].scatter(dx_arr, arr[0,0] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[0].scatter(dx_arr, arr[0,0] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[0].scatter(dx_arr, arr_ftcs[0,0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[0].scatter(dx_arr, arr_rk4[0,0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[0].scatter(dx_arr, arr_cn[0,0] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[0].legend(fontsize=body_size, loc="upper center")
    axs[0].set_ylabel(r'$\epsilon_{tot}$', fontsize=body_size)
    axs[0].set_xlabel(r'$dx$, $dy$', fontsize=body_size)

    # plot comp time vs dx at constant dt:
    axs[2].set_title(r'Computational time against spatial step size with $dt={0:.1e}$'.format(dt0), fontsize=body_size)
    
    if method=="ftcs":
        axs[2].scatter(dx_arr, arr[0,1] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[2].scatter(dx_arr, arr[0,1] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[2].scatter(dx_arr, arr[0,1] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[2].scatter(dx_arr, arr_ftcs[0,1] , label=r'FTCS scheme', color="black",  marker=".")
        axs[2].scatter(dx_arr, arr_rk4[0,1] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[2].scatter(dx_arr, arr_cn[0,1] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[2].legend(fontsize=body_size, loc="upper center")
    axs[2].set_ylabel(r'$T$ (s)', fontsize=body_size)
    axs[2].set_xlabel(r'$dx$, $dy$', fontsize=body_size)

    # plot cpu usage vs dx at constant dt:
    axs[4].set_title(r'CPU usage against spatial step size with $dt={0:.1e}$'.format(dt0), fontsize=body_size)
    
    if method=="ftcs":
        axs[4].scatter(dx_arr, arr[0,2] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[4].scatter(dx_arr, arr[0,2] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[4].scatter(dx_arr, arr[0,2] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[4].scatter(dx_arr, arr_ftcs[0,2] , label=r'FTCS scheme', color="black",  marker=".")
        axs[4].scatter(dx_arr, arr_rk4[0,2] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[4].scatter(dx_arr, arr_cn[0,2] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[4].legend(fontsize=body_size, loc="upper center")
    axs[4].set_ylabel(r'\% CPU ', fontsize=body_size)
    axs[4].set_xlabel(r'$dx$, $dy$', fontsize=body_size)

    # plot err vs dt at constant dx:
    axs[1].set_title(r'Total error against temporal step size with $dx=dy={0:.1e}$'.format(dx0), fontsize=body_size)
    
    if method=="ftcs":
        axs[1].scatter(dt_arr, arr[1,0] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[1].scatter(dt_arr, arr[1,0] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[1].scatter(dt_arr, arr[1,0] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[1].scatter(dt_arr, arr_ftcs[1,0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[1].scatter(dt_arr, arr_rk4[1,0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[1].scatter(dt_arr, arr_cn[1,0] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[1].legend(fontsize=body_size, loc="upper center")
    axs[1].set_ylabel(r'$\epsilon_{tot}$', fontsize=body_size)
    axs[1].set_xlabel(r'$dt$', fontsize=body_size)

    # plot comp time vs dx at constant dt:
    axs[3].set_title(r'Computational time against temporal step size with $dx=dy={0:.1e}$'.format(dx0), fontsize=body_size)
    
    if method=="ftcs":
        axs[3].scatter(dt_arr, arr[1,1] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[3].scatter(dt_arr, arr[1,1] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[3].scatter(dt_arr, arr[1,1] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[3].scatter(dt_arr, arr_ftcs[1,1] , label=r'FTCS scheme', color="black",  marker=".")
        axs[3].scatter(dt_arr, arr_rk4[1,1] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[3].scatter(dt_arr, arr_cn[1,1] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[3].legend(fontsize=body_size, loc="upper center")
    axs[3].set_ylabel(r'$T$ (s)', fontsize=body_size)
    axs[3].set_xlabel(r'$dt$', fontsize=body_size)

    # plot cpu usage vs dx at constant dt:
    axs[5].set_title(r'CPU usage against temporal step size with $dx=dy={0:.1e}$'.format(dx0), fontsize=body_size)
    
    if method=="ftcs":
        axs[5].scatter(dt_arr, arr[1,2] , label=r'FTCS scheme', color="black", marker=".")
    elif method=="rk4":
        axs[5].scatter(dt_arr, arr[1,2] , label=r'RK4 scheme', color="black", marker=".")
    elif method=="cn":
        axs[5].scatter(dt_arr, arr[1,2] , label=r'CN scheme', color="black",  marker=".")
    elif method=="all":
        axs[5].scatter(dt_arr, arr_ftcs[1,2] , label=r'FTCS scheme', color="black",  marker=".")
        axs[5].scatter(dt_arr, arr_rk4[1,2] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[5].scatter(dt_arr, arr_cn[1,2] , label=r'CN scheme', color="blue",  marker=".")        
    
    axs[5].legend(fontsize=body_size, loc="upper center")
    axs[5].set_ylabel(r'\% CPU ', fontsize=body_size)
    axs[5].set_xlabel(r'$dt$', fontsize=body_size)

    fig.savefig(os.path.join(out_dir, "precision2D.pdf"))
    fig.show()    

# visualise error vs runtime
def vis_err_v_T(method, dx0, dt0, dx_arr, dt_arr):

    # get data
    if method=="all":
        arr_ftcs, arr_rk4, arr_cn = run("caseA", method, dx0, dt0, dx_arr, dt_arr)
        arr_ftcs2d, arr_rk42d, arr_cn2d = run("caseB", method, dx0, dt0, dx_arr, dt_arr)
    else: 
        arr = run("caseA", method, dx0, dt0, dx_arr, dt_arr)
        arr2d = run("caseB", method, dx0, dt0, dx_arr, dt_arr)

    # set up figure
    fig, axs = plt.subplots(2,1,figsize = [18,10])
    fig.subplots_adjust(hspace = .5, wspace=.4) 
    axs = axs.ravel()  

    plt.suptitle(r'Error on Numerical Schemes Versus Runtime', fontsize=title_size)

    # 1D-case:
    axs[0].set_title(r'One-dimensional case', fontsize=body_size)
    
    if method=="ftcs":
        axs[0].scatter(arr[0,1], arr[0,0] , label=r'FTCS scheme', color="black", marker=".")
        axs[0].scatter(arr[1,1], arr[1,0] ,  color="black", marker=".")
    elif method=="rk4":
        axs[0].scatter(arr[0,1], arr[0,0] , label=r'RK4 scheme', color="black", marker=".")
        axs[0].scatter(arr[1,1], arr[1,0] ,  color="black", marker=".")
    elif method=="cn":
        axs[0].scatter(arr[0,1], arr[0,0] , label=r'CN scheme', color="black",  marker=".")
        axs[0].scatter(arr[1,1], arr[1,0] ,  color="black", marker=".")
    elif method=="all":
        axs[0].scatter(arr_ftcs[0,1], arr_ftcs[0,0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[0].scatter(arr_ftcs[1,1], arr_ftcs[1,0] ,  color="black",  marker=".")

        axs[0].scatter(arr_rk4[0,1], arr_rk4[0,0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[0].scatter(arr_rk4[1,1], arr_rk4[1,0] , color="gray",  marker=".")

        axs[0].scatter(arr_cn[0,1], arr_cn[0,0] , label=r'CN scheme', color="blue",  marker=".") 
        axs[0].scatter(arr_cn[1,1], arr_cn[1,0] , color="blue",  marker=".")         
    
    axs[0].legend(fontsize=body_size, loc="upper center")
    axs[0].set_ylabel(r'$\epsilon_{tot} ', fontsize=body_size)
    axs[0].set_xlabel(r'$T$ (s)', fontsize=body_size)

    # 2D-case:
    axs[1].set_title(r'Two-dimensional case', fontsize=body_size)
    
    if method=="ftcs":
        axs[1].scatter(arr[0,1], arr[0,0] , label=r'FTCS scheme', color="black", marker=".")
        axs[1].scatter(arr[1,1], arr[1,0] ,  color="black", marker=".")
    elif method=="rk4":
        axs[1].scatter(arr[0,1], arr[0,0] , label=r'RK4 scheme', color="black", marker=".")
        axs[1].scatter(arr[1,1], arr[1,0] ,  color="black", marker=".")
    elif method=="cn":
        axs[1].scatter(arr[0,1], arr[0,0] , label=r'CN scheme', color="black",  marker=".")
        axs[1].scatter(arr[1,1], arr[1,0] ,  color="black", marker=".")
    elif method=="all":
        axs[1].scatter(arr_ftcs[0,1], arr_ftcs[0,0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[1].scatter(arr_ftcs[1,1], arr_ftcs[1,0] ,  color="black",  marker=".")

        axs[1].scatter(arr_rk4[0,1], arr_rk4[0,0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[1].scatter(arr_rk4[1,1], arr_rk4[1,0] , color="gray",  marker=".")

        axs[1].scatter(arr_cn[0,1], arr_cn[0,0] , label=r'CN scheme', color="blue",  marker=".") 
        axs[1].scatter(arr_cn[1,1], arr_cn[1,0] , color="blue",  marker=".")         
    
    axs[1].legend(fontsize=body_size, loc="upper center")
    axs[1].set_ylabel(r'$\epsilon_{tot} ', fontsize=body_size)
    axs[1].set_xlabel(r'$T$ (s)', fontsize=body_size)


    fig.savefig(os.path.join(out_dir, "precision_err_v_time.pdf"))
    fig.show()

#####################
method="all"
case = "caseA"
dt0 = 0.001
dx0 = 0.1
dx_arr = [0.1, 0.2, 0.5, 0.01]
dt_arr = [0.001, 0.01, 0.05]    

vis_caseA(method, dx0, dt0, dx_arr, dt_arr)