import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def ftcs_1D(t0, t1, dt, x0, x1, dx, a, k_0=0, V=0):
    """Function calculates the forward centered difference scheme for 
        a one dimensional parabolic partial differential equation. 
    
    Args:
        t0: The starting time (generally zero) 
        t1: The end time
        dt: The time step size 
        
        x0: The start position of the x grid (generally zero) 
        x1: The end position of the x grid
        dx: The spacing between x grid points
        
        init_cond: The state of the function at a time, t=0
        a: Constant...
        k_0: Constant... optional argument set to 0 , if set to a number will become moving case
    
    Returns:
        f: Distribution function being solved as a function of time
        x: The position grid (for information)
        t: The time grid (for information)
    """
    
    #Initialise Arrays: 
    t = np.arange(t0, t1, dt) 
    x = np.arange(x0, x1, dx)
    n = len(t) 
    xn = len(x)
    
    #Initialise function at t = 0
    f = init_func(a,x,k_0=k_0)

    
    #Loop to compute FCTS scheme
    for i in range(1,n):
        f[1:xn-1] = f[1:xn-1] + dt*(1j/2)*(f[2:xn]-2*f[1:xn-1]+f[0:xn-2])/(dx**2)
    
    return(f,x,t)

def RK4_1D(t0, t1, dt, x0, x1, dx, a, k_0=0, V=0):
    """Function calculates the forward centered difference scheme for 
        a one dimensional parabolic partial differential equation. 
    
    Args:
        t0: The starting time (generally zero) 
        t1: The end time
        dt: The time step size 
        
        x0: The start position of the x grid (generally zero) 
        x1: The end position of the x grid
        dx: The spacing between x grid points
        
        init_cond: The state of the function at a time, t=0
        a: Constant...
        k_0: Constant... optional argument set to 0 , if set to a number will become moving case
    
    Returns:
        f: Distribution function being solved as a function of time
        x: The position grid (for information)
        t: The time grid (for information)
    """
    
    #Initialise Arrays: 
    t = np.arange(t0, t1, dt) 
    x = np.arange(x0, x1, dx)
    n = len(t) 
    xn = len(x)
    
    #Initialise function at t = 0
    f = init_func(a,x,k_0=k_0)

    
    #Loop to compute FCTS scheme
    for i in range(1,n):
       
        k1 = f[1:xn-1] + (1j/2)*(f[2:xn]-2*f[1:xn-1]+f[0:xn-2])/(dx**2)
        k2 =2
        k3 = 3
        k4 = 4
        f[0:xn] = f[0:xn] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    return(f,x,t)

def RK4(t0, t1, x0, x1, dx, dt, a, k_0):
    """ODE Solver for ODEs of the form dx/dt = kx using RK4. 
    Assumes that the initial condition is given at t = 0 
       
    Args: 
        t1: time to solve the equation up to
        x0: starting value of x, at t = 0
        k:  value of k in the ODE equation
        h:  step size of the method 
        
    Returns:
        t: An array in increments of h for the dimension of time up to t1
        x: An array in increments of h for the dimension of space up to x(t1)
    """
    
    #Initialise the arrays to be used
    # t is an array containing each of the timepoints that we will step forward to
    t = np.arange(t0,t1+dt,dt)
    # n is the number of timesteps in t
    n = np.shape(t)[0]
    # x starts as an empty array, but we will fill in the values we calculate in the loop, below
    x = np.arange(x0,x1+dx,dx) 
    xn = np.shape(x)[0]

    
    #Set the initial value of x (i.e. x[0])
    f = init_func(a,x,k_0=0)
    
    #Loop over the time values and calculate the derivative
    for i in range(1,n):
        k = (1j/2)*(f[2:xn]-2*f[1:xn-1]+f[0:xn-2])/(dx**2)
        k1 = dt*k*f[1:xn-1]
        k2 = dt*k*(f[1:xn-1]+(k1/2))
        k3 = dt*k*(f[1:xn-1]+(k2/2))
        k4 = dt*k*(f[1:xn-1]+k3)
        f[1:xn-1] = f[1:xn-1] + k1/6 + k2/3 + k3/3 + k4/6
    return(f, x, t)

def init_func(a,x,k_0=0):
    f_0 = (2*a/np.pi)**(1/4) * np.exp(-a*x**2) * np.exp(1j*k_0*x)
    return f_0

# def f_finite_difference(x,t)


a = 1
k_0 = 1

t0 = 0
t1 = 1
dt = 0.0001

x0 = -5
x1 = 5
dx = 0.1

f, x, t = RK4(t0, t1, x0, x1, dx, dt, a, k_0)
prob_density_func = f*np.conj(f)

plt.figure()
plt.plot(x,prob_density_func)
plt.ylim(0,1)
# plt.savefig("Group_test_RK4.png")
plt.show()

# for i in range(len(t1)):
#     f, x, t = ftcs_1D(t0, t1[i], dt, x0, x1, dx, a, k_0=k_0)

#     prob_density_func = f*np.conj(f)

#     plt.figure()
#     plt.plot(x,prob_density_func)
#     #plt.savefig("Group_test_{0}.png".format([i]))
#     plt.show()