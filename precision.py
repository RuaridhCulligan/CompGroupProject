#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     FUNCTIONS TO EVALUATE THE PRECISION OF NUMERICAL METHODS IN CASES A, B, C, D, E
#           

# import modules
import numpy as np
import matplotlib.pyplot as plt
import time 
import psutil
import os

# import functions
from NumericalMethods.log_handling import read_log
from NumericalMethods.ftcs import ftcs_1D, ftcs_2D, ftcs_2particle
from NumericalMethods.rk4 import rk4_1D, rk4_2D, rk4_2particle 
from NumericalMethods.rkf import rkf_1D, rkf_2D, rkf_2particle
from NumericalMethods.wavefuncs import an_sol_1D, an_sol_2D
from NumericalMethods.num_aux import integrate_1d, integrate_2d

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
t_end = 1 # 10
a     = 1 
kx0    = 1 
b     = 1 
ky0   = 1 
x0    = 0 
y0    = 0 
x_min = -20 
x_max = +20 
y_min = -20 
y_max = +20 
V0    = 100 
d     = 1 
w     = 1 
alpha = 1 

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
    sys_params[5] = V0
    sys_params[6] = d
    sys_params[7] = w
    sys_params[7] = alpha
    sys_params[9] = x0 
    sys_params[10] = y0

    # return in same format as "read_log":
    return case, settings, sys_params, num_par


# run simulation for a fixed dt, dx, dy and evaluate
def single_run(case, method, dt, dx, dy):

    case,  settings, sys_params, num_par = package(case, dx, dt, dy)

    pid = os.getpid()
    process = psutil.Process(pid)

    cpu_frac = process.cpu_percent()

    if case=="caseA":
        if method=="ftcs":
            start = time.time() 
            P, x, _, _ = ftcs_1D(case, settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x, _, _ = rk4_1D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rkf":
            start = time.time()
            P, x, _, _ = rkf_1D(case,  settings, sys_params, num_par) 
            stop = time.time()       

        P_an = np.abs(an_sol_1D(x,t_end, sys_params))**2
        P_diff = np.abs( P_an - P[0] )
        err = integrate_1d(P_diff, x)

    if case=="caseB":
        if method=="ftcs":
            start = time.time()
            P, x, y, _, _ = ftcs_2D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x, y, _, _ = rk4_2D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rkf":
            start = time.time()
            P, x, y, _, _ = rkf_2D(case,  settings, sys_params, num_par) 
            stop = time.time()     

        P_an =np.abs(an_sol_2D(x,y,t_end,sys_params))**2  
        P_diff = np.abs( P_an - P[0] )
        err = integrate_2d(P_diff, x, y)

    if case=="caseC":
        if method=="ftcs":
            start = time.time() 
            P, x, val, _ = ftcs_1D(case, settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x, val, _ = rk4_1D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rkf":
            start = time.time()
            P, x, val, _ = rkf_1D(case,  settings, sys_params, num_par) 
            stop = time.time()    

        err = np.abs(val[0] -1)

    if case=="caseD":
        if method=="ftcs":
            start = time.time() 
            P, x,y, val, _ = ftcs_2D(case, settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x,y, val, _ = rk4_2D(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rkf":
            start = time.time()
            P, x,y, val, _ = rkf_2D(case,  settings, sys_params, num_par) 
            stop = time.time()    

        err = np.abs(val[0] -1)    

    if case=="caseE":
        if method=="ftcs":
            start = time.time() 
            P, x, val, _ = ftcs_2particle(case, settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rk4":
            start = time.time()
            P, x, val, _ = rk4_2particle(case,  settings, sys_params, num_par) 
            stop = time.time()
        elif method=="rkf":
            start = time.time()
            P, x, val, _ = rkf_2particle(case,  settings, sys_params, num_par) 
            stop = time.time()    

        err = np.abs(val[0] -1)
    
    cpu_frac = process.cpu_percent()    

    return err, stop-start, cpu_frac

# for constant time step, loop through array of spatial step sizes (assuming dx=dy)
def space_loop(case, method, dt, dxdy_arr):

    err_arr     = np.empty(len(dxdy_arr))
    runtime_arr = np.empty(len(dxdy_arr))
    cpu_arr     = np.empty(len(dxdy_arr))
    arr         = np.empty(3, dtype="object")

    for i in np.arange(len(dxdy_arr)):
        dx, dy = dxdy_arr[i], dxdy_arr[i]
        err_arr[i], runtime_arr[i], cpu_arr[i] = single_run(case, method, dt, dx, dy)

    arr[0] = err_arr 
    arr[1] = runtime_arr 
    arr[2] = cpu_arr 

    return arr  

# for constant space step, loop through array of time step sizes (assuming dx=dy)
def time_loop(case, method, dxdy, dt_arr):

    err_arr     = np.empty(len(dt_arr))
    runtime_arr = np.empty(len(dt_arr))
    cpu_arr     = np.empty(len(dt_arr))
    arr         = np.empty(3, dtype="object")

    for i in np.arange(len(dt_arr)):
        dt = dt_arr[i]
        dx, dy = dxdy, dxdy
        err_arr[i], runtime_arr[i], cpu_arr[i] = single_run(case, method, dt, dx, dy)

    arr[0] = err_arr 
    arr[1] = runtime_arr 
    arr[2] = cpu_arr

    return arr       
  
# plot error against step size and computional time  
def err_vs_step_time(case,method,mode, d0, d_arr, fit ):

    # generate data
    if mode=="space_loop":
        
        d_const = "dt"
        if case != "caseB" and case != "caseD":
            d = "dx"
        else:
            d = "dx, dy"
        
        if method=="all":
            arr_ftcs = space_loop(case, "ftcs", d0, d_arr)
            arr_rk4 =  space_loop(case, "rk4", d0, d_arr)
            arr_rkf =   space_loop(case, "rkf", d0, d_arr)
        elif method=="ftcs" or method=="rk4" or method=="rkf":
            arr =  space_loop(case, method, d0, d_arr)  
    elif mode=="time_loop":
        
        d = "dt"
        if case != "caseB" and case != "caseD":
            d_const = "dx"
        else:
            d_const = "dx, dy"
        
        if method=="all":
            arr_ftcs = time_loop(case, "ftcs", d0, d_arr)
            arr_rk4 =  time_loop(case, "rk4", d0, d_arr)
            arr_rkf =   time_loop(case, "rkf", d0, d_arr)
        elif method=="ftcs" or method=="rk4" or method=="rkf":
            arr =  time_loop(case, method, d0, d_arr)

    # customise description in each case
    if case=="caseA":
        description = "free propagation in 1D"
    elif case=="caseB":
        description = "free propagation in 2D"
    elif case=="caseC":
        description = "a finite 1D potential barrier"
    elif case=="caseD":
        description = "single-slit diffraction in 2D" 
    elif case=="caseE":
        description = "two-particle collision in 2D"               

    # set up figure
    fig, axs = plt.subplots(2,1,figsize = [18,10])
    fig.subplots_adjust(hspace = .5, wspace=.4) 
    axs = axs.ravel()  

    plt.suptitle(r'Evaluation of numerical schemes for {0}'.format(description), fontsize=title_size)

    # plot error against step size
    if case=="caseA" or case=="caseB":
        axs[0].set_title(r'Error w.r.t. analytical solution against step size (with ${0}={1:.1e}$)'.format(d_const,d0), fontsize=body_size)
    else:
        axs[0].set_title(r'Error on normalisation against step size (with ${0}={1:.1e}$)'.format(d_const,d0), fontsize=body_size)    
    
    #fit function:
    if method != "all" and fit==True:
        z = np.polyfit(np.log(d_arr), np.log(arr[0]),1)
        p = np.poly1d(z)
        d_space = np.linspace(d_arr.min(), d_arr.max(), 100)

    if method=="ftcs":
        axs[0].scatter(d_arr, arr[0] , label=r'FTCS scheme', color="black", marker=".")
        if fit==True:
            axs[0].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\epsilon \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1])) 
    elif method=="rk4":
        axs[0].scatter(d_arr, arr[0] , label=r'RK4 scheme', color="black", marker=".")
        if fit==True:
            axs[0].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\epsilon \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1]))
    elif method=="rkf":
        axs[0].scatter(d_arr, arr[0] , label=r'RKF scheme', color="black",  marker=".")
        if fit==True:
            axs[0].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\epsilon \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1]))
    elif method=="all":
        axs[0].scatter(d_arr, arr_ftcs[0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[0].scatter(d_arr, arr_rk4[0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[0].scatter(d_arr, arr_rkf[0] , label=r'RKF scheme', color="blue",  marker=".")        
    
    axs[0].legend(fontsize=body_size, loc="upper center")
    axs[0].set_ylabel(r'$\epsilon$', fontsize=body_size)
    axs[0].set_xlabel(r'${0}$'.format(d), fontsize=body_size)

    # plot error against time

    #fit function:
    if method != "all" and fit==True:
        z = np.polyfit(np.log(arr[1]), np.log(arr[0]),1)
        p = np.poly1d(z)
        T_space = np.linspace(arr[1].min(), arr[1].max(), 100)

    if case=="caseA" or case=="caseB":
        axs[1].set_title(r'Error w.r.t. analytical solution against runtime (with ${0}={1:.1e}$)'.format(d_const,d0), fontsize=body_size)
    else:
        axs[1].set_title(r'Error on normalisation against runtime (with ${0}={1:.1e}$)'.format(d_const,d0), fontsize=body_size)
    
    if method=="ftcs":
        axs[1].scatter(arr[1], arr[0] , label=r'FTCS scheme', color="black", marker=".")
        if fit==True:
            axs[1].plot(T_space,T_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\epsilon \propto T^{{ {0:.4f} }}$'.format(p[1]))
    elif method=="rk4":
        axs[1].scatter(arr[1], arr[0] , label=r'RK4 scheme', color="black", marker=".")
        if fit==True:
            axs[1].plot(T_space,T_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\epsilon \propto T^{{ {0:.4f} }}$'.format(p[1]))
    elif method=="rkf":
        axs[1].scatter(arr[1], arr[0] , label=r'RKF scheme', color="black",  marker=".")
        if fit==True:
            axs[1].plot(T_space,T_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\epsilon \propto T^{{ {0:.4f} }}$'.format(p[1]))
    elif method=="all":
        axs[1].scatter(arr_ftcs[1], arr_ftcs[0] , label=r'FTCS scheme', color="black",  marker=".")
        axs[1].scatter(arr_rk4[1], arr_rk4[0] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[1].scatter(arr_rkf[1], arr_rkf[0] , label=r'RKF scheme', color="blue",  marker=".")        
    
    axs[1].legend(fontsize=body_size, loc="upper center")
    axs[1].set_ylabel(r'$\epsilon$', fontsize=body_size)
    axs[1].set_xlabel(r'$T(s)$', fontsize=body_size)

    # display and save figure
    fig.show()
    fig.savefig(os.path.join(out_dir, "Eps_vs_step_T.pdf"))
    
    return 0

# plot computational time and cpu usage against step size
def time_cpu_vs_step(case, method, mode, d0, d_arr, fit):

    # generate data
    if mode=="space_loop":
        
        d_const = "dt"
        if case != "caseB" and case != "caseD":
            d = "dx"
        else:
            d = "dx, dy"    
        
        if method=="all":
            arr_ftcs = space_loop(case, "ftcs", d0, d_arr)
            arr_rk4 =  space_loop(case, "rk4", d0, d_arr)
            arr_rkf =   space_loop(case, "rkf", d0, d_arr)
        elif method=="ftcs" or method=="rk4" or method=="rkf":
            arr =  space_loop(case, method, d0, d_arr)  
    elif mode=="time_loop":

        d = "dt"
        if case != "caseB" and case != "caseD":
            d_const = "dx"
        else:
            d_const = "dx, dy"

        if method=="all":
            arr_ftcs = time_loop(case, "ftcs", d0, d_arr)
            arr_rk4 =  time_loop(case, "rk4", d0, d_arr)
            arr_rkf =   time_loop(case, "rkf", d0, d_arr)
        elif method=="ftcs" or method=="rk4" or method=="rkf":
            arr =  time_loop(case, method, d0, d_arr)

    # customise description in each case
    if case=="caseA":
        description = "free propagation in 1D"
    elif case=="caseB":
        description = "free propagation in 2D"
    elif case=="caseC":
        description = "a finite 1D potential well"
    elif case=="caseD":
        description = "single-slit diffraction in 2D" 
    elif case=="caseE":
        description = "two-particle collision in 2D"               

    # set up figure
    fig, axs = plt.subplots(2,1,figsize = [18,10])
    fig.subplots_adjust(hspace = .5, wspace=.4) 
    axs = axs.ravel()  

    plt.suptitle(r'Evaluation of numerical schemes for {0}'.format(description), fontsize=title_size)

    # plot runtime against step size
    axs[0].set_title(r'Computational time against step size (with ${0}={1:.1e}$)'.format(d_const,d0), fontsize=body_size)

    #fit function:
    if method != "all" and fit==True:
        z = np.polyfit(np.log(d_arr), np.log(arr[1]),1)
        p = np.poly1d(z)
        d_space = np.linspace(d_arr.min(), d_arr.max(), 100)
    
    if method=="ftcs":
        axs[0].scatter(d_arr, arr[1] , label=r'FTCS scheme', color="black", marker=".")
        if fit==True:
            axs[0].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $T \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1])) 
    elif method=="rk4":
        axs[0].scatter(d_arr, arr[1] , label=r'RK4 scheme', color="black", marker=".")
        if fit==True:
            axs[0].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $T \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1])) 
    elif method=="rkf":
        axs[0].scatter(d_arr, arr[1] , label=r'RKF scheme', color="black",  marker=".")
        if fit==True:
            axs[0].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $T \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1])) 
    elif method=="all":
        axs[0].scatter(d_arr, arr_ftcs[1] , label=r'FTCS scheme', color="black",  marker=".")
        axs[0].scatter(d_arr, arr_rk4[1] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[0].scatter(d_arr, arr_rkf[1] , label=r'RKF scheme', color="blue",  marker=".")        
    
    axs[0].legend(fontsize=body_size, loc="upper center")
    axs[0].set_ylabel(r'$T (s)$', fontsize=body_size)
    axs[0].set_xlabel(r'${0}$'.format(d), fontsize=body_size)

    # plot cpu usage against step size
    axs[1].set_title(r'CPU usage against step size (with ${0}={1:.1e}$)'.format(d_const,d0), fontsize=body_size)
    
    #fit function:
    if method != "all" and fit==True:
        z = np.polyfit(np.log(d_arr), np.log(arr[2]),1)
        p = np.poly1d(z)
        d_space = np.linspace(d_arr.min(), d_arr.max(), 100)

    if method=="ftcs":
        axs[1].scatter(d_arr, arr[2] , label=r'FTCS scheme', color="black", marker=".")
        if fit==True:
            axs[1].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\% CPU \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1]))
    elif method=="rk4":
        axs[1].scatter(d_arr, arr[2] , label=r'RK4 scheme', color="black", marker=".")
        if fit==True:
            axs[1].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\% CPU \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1]))
    elif method=="rkf":
        axs[1].scatter(d_arr, arr[2] , label=r'RKF scheme', color="black",  marker=".")
        if fit==True:
            axs[1].plot(d_space,d_space**p[1]*np.exp(p[0]),color="red", ls="--" , label=r'best fit: $\% CPU \propto ({0})^{{ {1:.4f} }}$'.format(d, p[1]))
    elif method=="all":
        axs[1].scatter(d_arr, arr_ftcs[2] , label=r'FTCS scheme', color="black",  marker=".")
        axs[1].scatter(d_arr, arr_rk4[2] , label=r'RK4 scheme', color="gray",  marker=".")
        axs[1].scatter(d_arr, arr_rkf[2] , label=r'RKF scheme', color="blue",  marker=".")        
    
    axs[1].legend(fontsize=body_size, loc="upper center")
    axs[1].set_ylabel(r'$\% CPU$ ', fontsize=body_size)
    axs[1].set_xlabel(r'${0}$'.format(d), fontsize=body_size)

    # display and save figure
    fig.show()
    fig.savefig(os.path.join(out_dir, "T_CPU_vs_step.pdf"))

    return 0

# main function to run the code
def main(case, method, mode, d0, d_arr, fit=True):    

    if mode != "space_loop" and mode != "time_loop":
        print("Error: argument mode must be equal to 'space_loop' or 'time_loop' ")
        return 1
    else:
        err_vs_step_time(case,method, mode, d0, d_arr, fit)
        print("Done with 1/2, starting 2/2")
        time_cpu_vs_step(case,method, mode, d0, d_arr, fit) 
         



#####################
dt0 = 0.0001 #0.00001 
dx0 = 0.1
dx_arr = np.linspace(0.5, 0.1, 100)  # np.array([0.1, 0.2]) 
dt_arr = np.linspace(0.004, 0.006, 100) # RKF: np.linspace(0.001, 0.01, 100)  #  RK4: np.linspace(0.0001, 0.001, 100)       # FTCS: np.linspace(0.00001, 0.0001, 100)

main("caseA", "rk4", "time_loop", dx0, dt_arr, True)