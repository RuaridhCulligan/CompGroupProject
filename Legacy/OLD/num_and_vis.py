#-----------------------------------------------------------------------------
#
# "numpy_and_vis.py" - FUNCTIONS TO IMPLEMENT AND VISUALISE NUMERICAL SOLUTIONS 
# 
#   
#       Description: contains functions to numerically solve the Schrodinger 
#                    equation and display the results
#
#       Dependencies: standard modules (numpy, pandas, scipy); also requires
#                      "celluloid" which can be installed via pip with the
#                      command
#                                'pip3 install --upgrade celluloid'
#
#       Note: Output is based on parameter values in a log file of the
#             standarised format. The log file must contain lines of the
#             form "VARIABLE value". The VARIABLES appear in the same order
#             as listed below. Not all VARIABLES are relevant in each case. 
#             Arbitrary placeholder values can be used in this case which will
#             be ignored by the program. 
#
#                   CASE  (str):    can take on values "caseA", "caseB", "caseC", "caseD", 
#                                   "caseE"; each value corresponds to one of the systems
#                                   considered for the report (see Readme)
#                   METHOD (str):   can take on values "ftcs", "cn", "rk4", "all", "an"; each 
#                                   value correponds to the finite difference method to be 
#                                   used (or all)
#                   STATIC (float): can take on values "1" [to produce plot at t=T_END], 
#                                   "0" [to produce a GIF from t=0 to t=T_END], "0.5" [to produce plots for different times
#                                   between t=0 and t=T_END] 
#                   HIDE_A (float): can take on values "1" (hide analytical solution) or any
#                                   other float value to display analytical solution 
#                                   [only relevant for cases A and B]
#                   ADD_MET (str):  can take on values "no", "ftcs", "cn", "rk4"; each
#                                   value corresponds to an additional method for which output 
#                                   is to be displayed; 
#                   SHOW_V (float): if "1" display potential function; [not relevant for cases A,B]
#                   SAVE (float):   if "1" save output data to file, if else do not
#                   t_end (float):  end time of numerical solution 
#                   a  (float):     width of Gaussian wavepacket in the x-direction 
#                   k0 (float):     initial momentum of wavepacket in the x-direction
#                   b (float):      width of Gaussian wavepacket in the y-direction  
#                                   [only relevant for cases B and D]
#                   ky0 (float):    initial momentum of wavepacket in the y-direction
#                                   [only relevant for cases B and D]
#                   V0 (float):     potential amplitude [only relevant for cases C and E]
#                   d (float):      width of potential barrier in x-direction 
#                                   [only relevant for cases C and D]
#                   w (float):      width of potential barrier in y-direction
#                                   [only relevant for case D]
#                   alpha (float):  coefficient of inter-particle potential 
#                                   [only relevant for case E]
#                   x0 (float):     initial peak position of wavepacket
#                   y0 (float):     initial peak position of wavepacket in y direction
#                   x_min (float):  minimum value of x-grid
#                   x_max (float):  maximum value of x-grid
#                   dx (float):     x-grid step size
#                   dt (float):     t-grid step size
#                   y_min (float):  minimum value of y-grid
#                   y_max (float):  maximum value of y-grid
#                   dy (float):     y-grid step size
#
#             If no valid log file is found a default log file will be created 
#             by the program. 
#------------------------------------------------------------------------------

# import modules
import numpy as np
from celluloid import Camera
from scipy import integrate
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning) # surpress RuntimeWarnings 

# set standardised layout of plots
fig_dim    = [16, 8]   # dimensions
title_size = 16       # title font size
body_size  = 14       # axes and legends font size
tick_size  = 12       # tick mark font size 
plt.rcParams['text.usetex'] = True # enable LaTeX renadering
plt.rcParams['mathtext.fontset'] = 'cm' # use LateX font for maths
plt.rcParams['font.family'] = 'STIXGeneral' # use LateX font for text

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     SYSTEM FUNCTIONS RELATING TO WRITING / READING LOG FILES

# function to write a default log file in case no valid file is given
def create_log(path="log.txt"):
    
    f = open(path, "w")
    
    f.write("CASE caseA \n")
    f.write("METHOD ftcs \n")
    
    f.write("STATIC 1 \n")
    f.write("HIDE_A 0 \n")
    f.write("ADD_MET no \n")
    f.write("SHOW_V 0 \n")
    f.write("SAVE 0 \n")
    f.write("DIFF False")
    
    f.write("t_end 5 \n")
    f.write("a 1 \n")
    f.write("k0 1 \n")
    f.write("b 1 \n")
    f.write("ky0 1 \n")
    f.write("V0 5 \n")
    f.write("d 2 \n")
    f.write("w 2 \n")
    f.write("alpha 0.5 \n")
    f.write("x0 0 \n")
    f.write("y0 0 \n")
    
    f.write("x_min -50 \n")
    f.write("x_max +50 \n")
    f.write("dx 0.1 \n")
    f.write("dt 0.0001 \n")
    f.write("y_min -10 \n")
    f.write("y_max +10 \n")
    f.write("dy 0.0001 \n")
    
    
    f.close()

# read a log file at "path" and extract relecant info
def read_log(path):
    arr = np.loadtxt(path,dtype="str", delimiter=" ", usecols=1)
    
    # case, method as independent variables
    case     = arr[0]
    method   = arr[1]
    
    # variables corresponding to general simulation and output settings
    # stored in one array
    settings = np.array([float(arr[2]), float(arr[3]), arr[4], float(arr[5]),float(arr[6]), arr[7]])
    
    # variables corresponding to physical parameters in the simulation 
    # stored in one array
    sys_par  = arr[8:19].astype("float")
    
    # variables corresponding to grid parameters for the numerical solvers
    # stored in one array
    num_par = arr[19:].astype("float")

    
    # return results
    return case, method, settings, sys_par, num_par

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     AUXILIARY FUNCTIONS RELATING TO INITIAL / BOUNDARY CONDITIONS

# initialise wavepacket at position x and time t=0 (1D case)
def wavepacket_1d(x, sys_params):
    a =  sys_params[1]
    k0 = sys_params[2]

    x0 = sys_params[9]
    
    return (2*a/np.pi)**(1/4) * np.exp(-a*(x-x0)**2) * np.exp(1j*k0*(x-x0))

# initialise wavepacket at position x,y and time t=0 (2D case)
def wavepacket_2d(x,y,sys_params):
    
    a   = sys_params[1]
    kx0 = sys_params[2]
    b   = sys_params[3]
    ky0 = sys_params[4]

    x0 = sys_params[9]
    y0 = sys_params[10]
    
    f = np.zeros((len(x), len(y))).astype("complex")
    for i in np.arange(len(y)):
        f[i,0:] = (2*a/np.pi)**(1/4) * (2*b/np.pi)**(1/4) * np.exp(-a*(x[i]-x0)**2 - b*(y[0:]-y0)**2) * np.exp(1j*(kx0*(x[i]-x0) + ky0*(y[0:]-y0)))
        
    return f

# analytical solution for caseA (1D free wavepacket) at position x and time t
def an_sol_1D(x,t, sys_params):
    
    a  = sys_params[1]
    k0 = sys_params[2]

    x0 = sys_params[9]

    return 1/np.sqrt(1+2*a*t*1j) * (2*a/np.pi)**0.25 * np.exp(  (-a*(x-x0)**2 +1j * ( k0 * (x-x0) -0.5*k0**2*t) )  / (1+2*a*t*1j)  )
    
# analytical solution for caseB (2D free wavepacket)  at position x,y and time t
def an_sol_2D(x,y,t, sys_params):
    
    a   = sys_params[1]
    kx0 = sys_params[2]
    b   = sys_params[3]
    ky0 = sys_params[4]

    x0 = sys_params[9]
    y0 = sys_params[10]
    
    return 1/np.sqrt(1+2*a*t*1j) * (2*a/np.pi)**0.25 * np.exp(  (-a*(x-x0)**2 +1j * ( kx0 * (x-x0) -0.5*kx0**2*t) )  / (1+2*a*t*1j)  ) *1/np.sqrt(1+2*b*t*1j) * (2*b/np.pi)**0.25 * np.exp(  (-b*(y-y0)**2 +1j * ( ky0 * (y-y0) -0.5*ky0**2*t) )  / (1+2*b*t*1j)  )

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
    
    if np.abs(x) < 0.5*d or np.abs(y) < 0.5*w:
        return np.inf
    else: 
        return 0    

# potential function for case E at position x1,x2
def potential_E(x1,x2, sys_params):
    
    V0    = sys_params[5]
    alpha = sys_params[8]
    
    return V0*np.exp(-alpha*np.abs(x1-x2)**2)      

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     NUMERICAL SOLVERS


# check normalisation: numerically integrate 'func' and return result;
def integrate_1d(func, x_vals):

    N = len(func) -1        # number of sub-intervals
    h = x_vals[1]-x_vals[0] # x-spacing 

    # ensure N is even (pad zeros if need be):
    if N % 2 == 1:
        func   = np.append(func, func[0])
        x_vals = np.append(x_vals, x_vals.min)

    # calculate integral as sum:
    I = h /3 * (func[0]+ func[-1]) 

    for i in np.arange(1, N/2):
        I += h/3 * 4* func[int(2*i-1)] + h/3 * 2 * func[int(2*i)]

    I += h/3 * 4* func[-2]    

    return I

# check normalisation in 2D: numerically integrate 'func' and return result;
def integrate_2d(func, x_vals,y_vals):

    # integrate func with respect to y:
    I_y = np.empty(len(x_vals))
    for i in np.arange(len(x_vals)):
        I_y[i] = integrate_1d(func[i,:], y_vals)

    # integrate with respect to x
    I = integrate_1d(I_y, x_vals)
    
    return I

# tridiagonal matrix solver to solve tridiagonal system of equations
def tridiag_solv(Mat1, Mat2, vec):

    rhs = np.matmul(Mat2,vec)

    n = rhs.shape
    # c and d are just intermediate constants
    c = np.zeros(n)
    d = np.zeros(n)

    # vector of the solution values
    sol_vec = np.zeros(n)

    # calculating constants
    c[0] = Mat1[1,0]/Mat1[0,0]
    d[0] = rhs[0]/Mat1[0,0]
    c[1:n-1] = Mat1[2:n,1:n-1]/(Mat1[1:n-1,1:n-1] - Mat1[1:n-1,2:n]*c[0:n-2])
    d[1:n-1] = (rhs[1:n-1] - Mat1[1:n-1,2:n]*d[0:n-2])/(Mat1[1:n-1,1:n-1] - Mat1[1:n-1,2:n]*c[0:n-2])

    # calculating solution vector
    sol_vec[n-1] = (rhs[n-1] - Mat1[n-1,n]*d[n-1])/(Mat1[n-1,n-1] - Mat1[n-1,n]*c[n-2])
    sol_vec[0:n-1] = d[0:n-1] - c[0:n-1]*sol_vec[1:n]

    return sol_vec


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
            psi[0:,0] = 0
            psi[0:, yn-1] = 0
            psi[xn-1,0:] = 0
            psi[0, 0:] = 0
            
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
     
    #Loop over the time values and calculate the derivative
    for i in range(1,tn):
        f = np.zeros(xn).astype("complex"); k1 = np.zeros(xn).astype("complex");  k2 = np.zeros(xn).astype("complex"); k3 = np.zeros(xn).astype("complex"); k4 = np.zeros(xn).astype("complex")

        f[1:xn-1] = (1j/2)*(psi[0:xn-2]-2*psi[1:xn-1]+psi[2:xn])/(dx**2) + V[1:xn-1]*psi[1:xn-1]

        k1 = f

        k2[1:xn-1] = f[1:xn-1] + (1j/4)*(k1[0:xn-2]-2*k1[1:xn-1]+k1[2:xn])/(dx**2) + V[1:xn-1]*(psi[1:xn-1] + k1[1:xn-1])

        k3[1:xn-1] = f[1:xn-1] + (1j/4)*(k2[0:xn-2]-2*k2[1:xn-1]+k2[2:xn])/(dx**2) + V[1:xn-1]*(psi[1:xn-1] + k2[1:xn-1])

        k4[1:xn-1] = f[1:xn-1] + (1j/2)*(k3[0:xn-2]-2*k3[1:xn-1]+k3[2:xn])/(dx**2) + V[1:xn-1]*(psi[1:xn-1] + k2[1:xn-1])

        psi = psi + dt*(k1/6 + k2/3 + k3/3 + k4/6)

        #Force boundary conditions on the off chance something has gone wrong and they contain a value
        psi[0] = 0; psi[xn-1] = 0

        if (t[i] in T):
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

        if (t[k] in T):
            P[j]   = np.abs(psi)**2
            val[j] = integrate_2d(P[j],x,y)
            j += 1   
                                                 
    # return output                                    
    return P, x, y, val, T

# RK4 method to solve two-particle case numerically (case E)  
def rk4_2particle(case, settings, sys_par, num_par): 
    
    """
    to do!
    """
    
    return 0

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
    sigma = np.ones(xn)*(dt*1j)/(2*dx**2)

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

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     VISUALISATION FUNCTIONS

# visualise solution in one-dimensional case (cases A & C)
def visualise_1D(case,method, settings, sys_par, num_par):

    diff = settings[5]

    # compute relevant numerical solutions
    ADD_MET = settings[2]
    if method=="ftcs" and ADD_MET == "no":
        P, x, val, T = ftcs_1D(case, settings, sys_par, num_par)
    elif method=="rk4" and ADD_MET == "no":
        P, x, val, T = rk4_1D(case, settings, sys_par, num_par)
    elif method=="cn" and ADD_MET == "no":
        P, x, val, T = cn_1D(case, settings, sys_par, num_par)
    elif method=="all" and ADD_MET == "no":
        P_ftcs, x, val_ftcs, T = ftcs_1D(case, settings, sys_par, num_par)
        P_rk4, x, val_rk4, T   = rk4_1D(case, settings, sys_par, num_par)
        P_cn, x, val_cn, T     = cn_1D(case, settings, sys_par, num_par)
    elif method=="an":
        x_min   = num_par[0]
        x_max   = num_par[1]
        dx      = num_par[2]
        t_start = 0
        t_end   = sys_par[0]
        dt      = num_par[3]
        x = np.arange(x_min, x_max+dx, dx)

        if settings[2]=="0":    
            T   = np.arange(t_start, t_end+dt, dt*100)
        if settings[2]=="0.5": 
            T   = np.array([t_start,t_end/8,t_end/4,t_end/2, 3*t_end/4, t_end]) 
        else:
            T   = np.array([t_end])

        P = np.empty(len(T), dtype="object")
        for i in np.arange(len(T)):
            psi = an_sol_1D(x, T[i], sys_par)
            P[i] = np.abs(psi)**2

    
    # implement option to compute two numerical solutions using
    # the variable ADD_MET in the log file
    
    elif (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):
        P_ftcs, x, val_ftcs, T = ftcs_1D(case, settings, sys_par, num_par)
        P_rk4, x, val_rk4, T = rk4_1D(case, settings, sys_par, num_par)
    elif (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
        P_rk4, x, val_ftcs, T = rk4_1D(case, settings, sys_par, num_par)
        P_cn, x, val_cn, T = cn_1D(case, settings, sys_par, num_par)
    elif (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
        P_ftcs, x, val_ftcs, T = ftcs_1D(case, settings, sys_par, num_par)
        P_cn, x, val_cn, T = cn_1D(case, settings, sys_par, num_par)
    
    # implement option to display potential in relevant cases
    SHOW_V = float(settings[3])
    v = False
    V0 = sys_par[5]
    
    if SHOW_V==1 and case=="caseC":
            d = sys_par[6]
            V_x = np.array([d/2, d/2])
            v = True
            
    # compute analytical solution in relevant cases:
    an = False
    
    if (case=="caseA" and float(settings[1])!=1) or diff == "True":
        an = True
        
        if method != "all" and ADD_MET == "no":
            P_an = np.copy(P)
        elif method=="all":
            P_an = np.copy(P_ftcs)
        elif ADD_MET=="ftcs":
            P_an = np.copy(P_ftcs) 
        elif ADD_MET=="rk4":
            P_an = np.copy(P_rk4)  
        elif ADD_MET=="cn":
            P_an = np.copy(P_cn)           
      
        for i in np.arange(len(T)):
            psi = an_sol_1D(x, T[i], sys_par)
            P_an[i] = np.abs(psi)**2
        
    # produce visualisation in the static case:
    if float(settings[0])==1:
        plt.figure(figsize=fig_dim) 
        
        if case=="caseA":
            plt.title(r'Free propagation of a Gaussian wavepacket (at $t={0:.3f})$'.format(sys_par[0]), fontsize=title_size)
        elif case=="caseC":
            plt.title(r'Tunneling of a Gaussian wavepacket (at $t={0:.3f})$'.format(sys_par[0]), fontsize=title_size)
        
        plt.ylabel(r'Probability density $|\Psi(x,t)|^2$', fontsize=body_size)
        plt.xlabel(r'Spatial dimension $x$', fontsize=body_size)
        
        if method=="ftcs" and ADD_MET == "no":
            if diff == "True" and case == "caseA":
                P_diff = np.abs(P - P_an)
                plt.plot(x,P_diff[0],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff[0],x)))
            else:
                plt.plot(x,P[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val[0]))

            if v==True:
                plt.plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")

        elif method=="rk4" and ADD_MET == "no": 
            if diff == "True" and case == "caseA":
                P_diff = np.abs(P - P_an)
                plt.plot(x,P_diff[0],color="black", label=r'Error on RK4 scheme (total: {0:.3f})'.format(integrate_1d(P_diff[0],x)))
            else:
                plt.plot(x,P[0],color="black", label=r'RK4 method normalised to {0:.4f} '.format(val[0]))

            if v==True:
                plt.plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
       
        elif method=="cn" and ADD_MET == "no": 
            if diff == "True" and case == "caseA":
                P_diff = np.abs(P - P_an)
                plt.plot(x,P_diff[0],color="black", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff[0],x)))
            else:
                plt.plot(x,P[0],color="black", label=r'CN scheme normalised to {0:.4f} '.format(val[0]))

            if v==True:
                plt.plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
        
        elif method=="all" and ADD_MET == "no":
            if diff == "True" and case == "caseA":
                P_diff1 = np.abs(P_ftcs - P_an)
                P_diff2 = np.abs(P_rk4 - P_an)
                P_diff3 = np.abs(P_cn - P_an)
                plt.plot(x,P_diff1[0],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff1[0],x)))
                plt.plot(x,P_diff2[0],color="gray", label=r'Error on RK4 scheme (total: {0:.3f})'.format(integrate_1d(P_diff2[0],x)))
                plt.plot(x,P_diff3[0],color="blue", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff3[0],x)))
            else:
                plt.plot(x,P_ftcs[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
                plt.plot(x,P_rk4[0],color="grey", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
                plt.plot(x,P_cn[0],color="blue", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
                
            if v==True:
                plt.plot(V_x,np.array([0,P_ftcs[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P_ftcs[0].max()]),color="green",linestyle="--")

        elif method=="an" and ADD_MET=="no":
            plt.plot(x,P[0],color="black", label=r'Analytical solution normalised to {0:.4f}'.format(integrate_1d(P[0],x)))

            if v==True:
                plt.plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
        

        # cases for displaying two numerical solutions using ADD_MET
        elif (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):
            if diff == "True" and case == "caseA":
                P_diff1 = np.abs(P_ftcs - P_an)
                P_diff2 = np.abs(P_rk4 - P_an)
                plt.plot(x,P_diff1[0],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff1[0],x)))
                plt.plot(x,P_diff2[0],color="gray", label=r'Error on RK4 scheme (total: {0:.3f})'.format(integrate_1d(P_diff2[0],x)))
            else:
                plt.plot(x,P_ftcs[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
                plt.plot(x,P_rk4[0],color="grey", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
                

            if v==True:
                plt.plot(V_x,np.array([0,P_ftcs[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P_ftcs[0].max()]),color="green",linestyle="--")
        
        elif (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
            if diff == "True" and case == "caseA":
                P_diff2 = np.abs(P_rk4 - P_an)
                P_diff3 = np.abs(P_cn - P_an)
                plt.plot(x,P_diff2[0],color="black", label=r'Error on RK4 scheme (total: {0:.3f})'.format(integrate_1d(P_diff2[0],x)))
                plt.plot(x,P_diff3[0],color="gray", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff3[0],x)))
            else:
                plt.plot(x,P_rk4[0],color="grey", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
                plt.plot(x,P_cn[0],color="blue", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0])) 

            if v==True:
                plt.plot(V_x,np.array([0,P_cn[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P_cn[0].max()]),color="green",linestyle="--")
        
        elif (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
            if diff == "True" and case == "caseA":
                P_diff1 = np.abs(P_ftcs - P_an)
                P_diff3 = np.abs(P_cn - P_an)
                plt.plot(x,P_diff1[0],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff1[0],x)))
                plt.plot(x,P_diff3[0],color="gray", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff3[0],x)))
            else:
                plt.plot(x,P_ftcs[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
                plt.plot(x,P_cn[0],color="blue", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
                        
            if v==True:
                plt.plot(V_x,np.array([0,P_ftcs[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                plt.plot(-V_x,np.array([0,P_ftcs[0].max()]),color="green",linestyle="--")

        if an==True and method != "an" and diff == "False":
            plt.plot(x,P_an[0],color="red",linestyle="--", label=r'Analytical solution')
        
        plt.legend(fontsize=body_size, loc="upper right")
        plt.savefig("visualisation.pdf")
        plt.show()
 
    # produce visualisation in the non-static (GIF) case:
    if float(settings[0])==0:
        
        fig=plt.figure(figsize=fig_dim)
        camera = Camera(fig)

        if case=="caseA":
            plt.title(r'Free propagation of a Gaussian wavepacket', fontsize=title_size)
        elif case=="caseC":
            plt.title(r'Tunneling of a Gaussian wavepacket', fontsize=title_size)

        for i in np.arange(len(T)):
            """
            do this!
            """
            camera.snap()
        

        animation = camera.animate()
        animation.save("visualisation.gif")
    
    # produce visualisation in the semi-static (subplot) case:
    if float(settings[0])==0.5:
        
        t_arr = [0,5,10,20]
        fig, axs = plt.subplots(2,2, figsize=[15,8])
        if ADD_MET == "no":
            fig.suptitle("Semistatic case for case {0} using {1} method".format(case[-1], method),fontsize="16")
            fig.subplots_adjust(hspace = .25, wspace=.1)
            axs = axs.ravel()

            for i in range(len(t_arr)):
                axs[i].set_title("t={0}".format(t_arr[i]))
                axs[i].plot(x,P[t_arr[i]],color="black", label=method +" scheme normalised to "+str(val[0]))
                axs[i].legend()
                axs[i].set_ylabel("f(x)")
                axs[i].set_xlabel("x")

        elif method == "all":
            fig.suptitle("Semistatic case for case {0} using RK4, FTCS, and Crank-Nicolson  methods".format(case[-1]),fontsize="16")
            fig.subplots_adjust(hspace = .25, wspace=.1)
            axs = axs.ravel()

            for i in range(len(t_arr)):
                axs[i].set_title("t={0}".format(t_arr[i]))
                axs[i].plot(x,P_ftcs[t_arr[i]],color="black", label=r'FTCS scheme normalised to '.format(val_ftcs[0]))
                axs[i].plot(x,P_rk4[t_arr[i]],color="gray", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
                axs[i].plot(x,P_cn[t_arr[i]],color="blue", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
                axs[i].legend()
                axs[i].set_ylabel("f(x)")
                axs[i].set_xlabel("x")

        else:
            if (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):
                fig.suptitle("Semistatic case for case {0} using FTCS and RK4 method".format(case[-1]),fontsize="16")
                fig.subplots_adjust(hspace = .25, wspace=.1)
                axs = axs.ravel()
                for i in range(len(t_arr)):
                    axs[i].set_title("t={0}".format(t_arr[i]))
                    axs[i].plot(x,P_ftcs[t_arr[i]],color="black", label=r'FTCS scheme normalised to FIX ')
                    axs[i].plot(x,P_rk4[t_arr[i]],color="gray", label=r'RK4 scheme normalised to FIX ')
                    axs[i].legend()
                    axs[i].set_ylabel("f(x)")
                    axs[i].set_xlabel("x")
            if (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
                
                fig.suptitle("Semistatic case for case {0} using RK4 and CN methods".format(case[-1]),fontsize="16")
                fig.subplots_adjust(hspace = .25, wspace=.1)
                axs = axs.ravel()
                for i in range(len(t_arr)):
                    axs[i].set_title("t={0}".format(t_arr[i]))
                    axs[i].plot(x,P_rk4[t_arr[i]],color="black", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
                    axs[i].plot(x,P_cn[t_arr[i]],color="gray", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
                    axs[i].set_ylabel("f(x)")
                    axs[i].set_xlabel("x")

            if (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
                
                fig.suptitle("Semistatic case for case {0} using FTCS and CN methods".format(case[-1]),fontsize="16")
                fig.subplots_adjust(hspace = .25, wspace=.1)
                axs = axs.ravel()
                for i in range(len(t_arr)):
                    axs[i].set_title("t={0}".format(t_arr[i]))
                    axs[i].plot(x,P_ftcs[t_arr[i]],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
                    axs[i].plot(x,P_cn[t_arr[i]],color="gray", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
                    axs[i].legend()
                    axs[i].set_ylabel("f(x)")
                    axs[i].set_xlabel("x")
        plt.show()
        
     # add option to save some of the numerical output to file
        
        """
        to do!
        """

    return 0


# visualise solution in two-dimensional case (cases B & D)
def visualise_2D(case,method, settings, sys_par, num_par):
    
    # compute relevant numerical solutions
    ADD_MET = settings[2]
    if method=="ftcs" and ADD_MET == "no":
        P, x, y , val, T = ftcs_2D(case, settings, sys_par, num_par)
    if method=="rk4" and ADD_MET == "no":
        P, x, y, val, T = rk4_2D(case, settings, sys_par, num_par)
    if method=="cn" and ADD_MET == "no":
        P, x, y, val, T = cn_2D(case, settings, sys_par, num_par)
    if method=="all" and ADD_MET == "no":
        P_ftcs, x, y, val_ftcs, T = ftcs_2D(case, settings, sys_par, num_par)
        P_rk4, x, y, val_rk4, T   = rk4_2D(case, settings, sys_par, num_par)
        P_cn, x, y, val_cn, T     = cn_2D(case, settings, sys_par, num_par)
    

    # implement option to compute two numerical solutions using
    # the variable ADD_MET in the log file
    
    if (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):
        P_ftcs, x, y, val_ftcs, T = ftcs_2D(case, settings, sys_par, num_par)
        P_rk4, x, y, val_rk4, T = rk4_2D(case, settings, sys_par, num_par)
    if (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
        P_rk4, x, y, val_ftcs, T = rk4_2D(case, settings, sys_par, num_par)
        P_cn, x, y, val_cn, T = cn_2D(case, settings, sys_par, num_par)
    if (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
        P_ftcs, x, y, val_ftcs, T = ftcs_2D(case, settings, sys_par, num_par)
        P_cn, x, y, val_cn, T = cn_2D(case, settings, sys_par, num_par)

    # set up meshgrid with x,y values
    X, Y = np.meshgrid(x,y)    
    
    # implement option to display potential in relevant cases
    SHOW_V = float(settings[3])
    v  = False
    V0 = sys_par[5]
    d  = sys_par[6]
    w  = sys_par[7] 
    
    if SHOW_V==1 and case=="caseD":
            
            """
            do later
            """
            
    # compute analyticquial solution in relevant cases:
    an = False
    
    if case=="caseB" and float(settings[1])!=1:
        an = True
        
        if method != "all" and ADD_MET == "no":
            P_an = np.copy(P)
        elif method=="all":
            P_an = np.copy(P_ftcs)
        elif ADD_MET=="ftcs":
            P_an = np.copy(P_ftcs) 
        elif ADD_MET=="rk4":
            P_an = np.copy(P_rk4)  
        elif ADD_MET=="cn":
            P_an = np.copy(P_cn)           
      
        for i in np.arange(len(T)):
            psi = an_sol_2D(X,Y, T[i], sys_par)
            P_an[i] = np.abs(psi)**2
        
    # produce visualisation in the static case:
    if float(settings[0])==1:
        plt.figure(figsize=fig_dim) 
        ax = plt.axes(projection='3d')
        
        if case=="caseB":
            ax.set_title(r'Free propagation of a Gaussian wavepacket (at $t={0:.3f})$'.format(sys_par[0]), fontsize=title_size)
        elif case=="caseD":
            ax.set_title(r'Gaussian wavepacket in an infinite potential well (at $t={0:.3f})$'.format(sys_par[0]), fontsize=title_size)
        
        ax.set_zlabel(r'Probability density $|\Psi(x,y,t)|^2$', fontsize=body_size)
        ax.set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
        ax.set_ylabel(r'Spatial dimension $y$', fontsize=body_size)
        
        if method=="ftcs" and ADD_MET == "no": 
            ax.plot_wireframe(X,Y,P[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val[0]))

            if v==True:
                """ """

        if method=="rk4" and ADD_MET == "no": 
            plt.plot(x,P[0],color="black", label=r'RK4 scheme normalised to {0:.4f} '.format(val[0]))

            if v==True:
                """ """  
       
        if method=="cn" and ADD_MET == "no": 
            plt.plot(x,P[0],color="black", label=r'CN scheme normalised to {0:.4f} '.format(val[0]))

            if v==True:
                """ """
        
        if method=="all" and ADD_MET == "no":
            plt.plot(x,P_ftcs[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
            plt.plot(x,P_rk4[0],color="grey", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
            plt.plot(x,P_cn[0],color="blue", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))

            if v==True:
                """ """
        if an==True:
            ax.plot_wireframe(X,Y,P_an[0],color="red",linestyle="--", label=r'analytical solution') 
        

        # cases for displaying two numerical solutions using ADD_MET
        if (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):
            plt.plot(x,P_ftcs[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
            plt.plot(x,P_rk4[0],color="gray", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
            
            if v==True:
                """ """
        
        if (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
            plt.plot(x,P_rk4[0],color="black", label=r'RK4 scheme normalised to {0:.4f} '.format(val_rk4[0]))
            plt.plot(x,P_cn[0],color="gray", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
            
            if v==True:
                """ """
        
        if (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
            plt.plot(x,P_ftcs[0],color="black", label=r'FTCS scheme normalised to {0:.4f} '.format(val_ftcs[0]))
            plt.plot(x,P_cn[0],color="gray", label=r'CN scheme normalised to {0:.4f} '.format(val_cn[0]))
            
            if v==True:
                """ """

        plt.legend(fontsize=body_size, loc="upper right")
        plt.savefig("visualisation.pdf")
        plt.show()
 
    # produce visualisation in the non-static (GIF) case:
        """
         to do! 
        """
    
    # produce visualisation in the semi-static (subplot) case:
        
        """
        to do! 
        """
        
     # add option to save some of the numerical output to file
        
        """
        to do!
        """

    
    return 0

# visualise solution in two-particle case (case E)
def visualise_2particle(case, method, settings, sys_par, num_par):
    
    """
    to do!
    """
    
    return 0



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     EXECUTION FUNCTIONS AND FLOW CONTROL

def main(log_file="log.txt"):
    
    # if log file does not exist in location create log file with default values
    if os.path.exists(log_file)==False:
        create_log(log_file)
    
    # extract information from log files
    case, method, settings, sys_par, num_par=read_log(log_file)
    
    if case=="caseA" or case=="caseC":
        visualise_1D(case,method, settings, sys_par, num_par)
    if case=="caseB" or case=="caseD":
        visualise_2D(case,method, settings, sys_par, num_par)
    if case=="caseE":
        visualise_2particle(case,method, settings, sys_par, num_par)    
                                                   

main()

#create_log("log.txt")