import numpy as np
import scipy as sc
from scipy import integrate
import matplotlib.pyplot as plt

# Using this space so we can comment current shit going on in the program
#
# Dillon - Need to comment on functions, do summaries and update variable names. Overall tidy up functions
# Got an idea for RK4 to fix so will impliment that and fix that hopefully
# Looked into more efficient ways of for looping, will look into that more and impliment
# Try impliment potential into ftcs as RK4 and Crank nichelson should already work with potentials.
# Will test potentials after tuesday meeting though
# Once true analytic solution is avaliable can tweak shit
# Change all matrices and vectors to complex

def ftcs(t0, t1, dt, x0, x1, dx,  a, V=None, Y=None, k_0=0, ky0=0, y0=0, y1=0, dy=0, b=0):
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
    if Y.type == None or Y == 1:
        #Initialise Arrays: 
        t = np.arange(t0, t1, dt) 
        x = np.arange(x0, x1, dx)
        n = len(t) 
        xn = len(x)
        
        #Initialise function at t = 0
        f = init_func_1D(a,x,kx0=0)
        f.dtype = np.complex
        
        #Loop to compute FCTS scheme
        for i in range(1,n):
            f[1:xn-1] = f[1:xn-1] + dt*1j*((f[2:xn]-2*f[1:xn-1]+f[0:xn-2])/(2*dx**2) + V[1:xn-1]*f[1:xn-1]) 
            f[0] = 0
            f[xn-1] = 0
    elif Y == 2:
        #Initialise Arrays: 
        t = np.arange(t0, t1, dt) 
        x = np.arange(x0, x1, dx)
        y = np.arange(y0, y1, dy)
        n = len(t) 
        xn = len(x)
        yn = len(y)
        
        #Initialise function at t = 0
        f = init_func_2D(a,b,x,y,kx0=0,ky0=0)
        f.dtype = np.complex
        
        #Loop to compute FCTS scheme
        for i in range(1,n):
            for i in range(1,xn-1):
                f[i,1:yn-1] = f[i,1:yn-1] + dt*1j*((f[i+1,2:yn]-2*f[i,1:yn-1]+f[i-1,0:yn-2])/(2*dx**2) + V[i,1:yn-1]*f[i,1:yn-1]) 
                f[:,0] = 0
                f[:, yn] = 0
                f[xn,:] = 0
                f[:, :] = 0
        
        return(f,x,t)


#After some more reaserch we need to impliment method of lines for this to work
#Will delete this comment if I complete it xoxo
def RK4(t0, t1, dt, x0, x1, dx, a, V, k_0=0):
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
    phi = init_func_1D(a,x,k_0=k_0)
    
    #Loop over the time values and calculate the derivative
    for i in range(1,n):
        f = np.zeros(xn, dtype=np.complex); k1 = np.zeros(xn,dtype=np.complex);  k2 = np.zeros(xn, dtype=np.complex); k3 = np.zeros(xn, dtype=np.complex); k4 = np.zeros(xn, dtype=np.complex)

        f[1:xn-1] = (1j/2)*(phi[0:xn-2]-2*phi[1:xn-1]+phi[2:xn])/(dx**2)# + V[1:xn-1]*phi[1:xn-1]

        k1 = f

        k2[1:xn-1] = f[1:xn-1] + (1j/4)*(k1[0:xn-2]-2*k1[1:xn-1]+k1[2:xn])/(dx**2)# + V[1:xn-1]*(phi[1:xn-1] + k1[1:xn-1])

        k3[1:xn-1] = f[1:xn-1] + (1j/4)*(k2[0:xn-2]-2*k2[1:xn-1]+k2[2:xn])/(dx**2)# + V[1:xn-1]*(phi[1:xn-1] + k2[1:xn-1])

        k4[1:xn-1] = f[1:xn-1] + (1j/2)*(k3[0:xn-2]-2*k3[1:xn-1]+k3[2:xn])/(dx**2)# + V[1:xn-1]*(phi[1:xn-1] + k2[1:xn-1])

        phi = phi + dt*(k1/6 + k2/3 + k3/3 + k4/6)

        #Force boundary conditions on the off chance something has gone wrong and they contain a value
        phi[0] = 0; phi[xn-1] = 0

    return(phi, x, t)

def crank_nichelson(t0, t1, dt, x0, x1, dx, a, k_0=0, V=0):
    """Function calculates the crank nichelson scheme for 
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
        V: Potential function, optional argument that is initially set to 0
    
    Returns:
        f: Distribution function being solved as a function of time
        x: The position grid (for information)
        t: The time grid (for information)
    """
    #Initialise the arrays to be used
    # t is an array containing each of the timepoints that we will step forward to
    t = np.arange(t0,t1+dt,dt)
    tn = np.shape(t)[0]

    # x starts as an empty array, but we will fill in the values we calculate in the loop, below
    x = np.arange(x0,x1+dx,dx) 
    xn = np.shape(x)[0]

    sigma = np.ones(xn)*(dt*1j)/(4*dx**2)

    A = np.diag(-sigma[0:xn-1], 1) + np.diag(1+2*sigma) + np.diag(-sigma[0:xn-1], -1)
    B = np.diag(sigma[0:xn-1], 1) + np.diag(1-2*sigma+(V*dt*1j)) + np.diag(sigma[0:xn-1], -1)
    
    #Set the initial value of x (i.e. x[0])
    phi = init_func_1D(a,x,k_0=k_0)
    phi.dtype = np.complex

    for i in range(1,tn):
        phi = np.linalg.solve(A, B.dot(phi))
    return phi, x, t

def init_func_1D(a,x,k_0=0):
    f0 = (2*a/np.pi)**(1/4) * np.exp(-a*x**2) * np.exp(1j*k_0*x)
    return f0

def init_func_2D(a,b,x,y,kx0=0,ky0=0):
    f0 = np.zeros(len(x),len(y))
    for i in range(len(y)):
        f0[i,0:] = (2*a/np.pi)**(1/4) * (2*b/np.pi)**(1/4) * np.exp(-a*x[i]**2 - b*y[0:]**2) * np.exp(1j*(kx0*x[i] + ky0*y[0:]))
    return f0

def schrodinger_sol_analytical_1D(x, t, a, k0):
    f = (1 / (8*a*np.pi))**(1/4)*np.exp(-k0**2 /(4*a)) * (1/(4*a) + 1j*t/2)**(-1/2) * np.exp(((k0/(2*a) + 1j*x)**2) / (a + 2*1j*t))
    return f

def schrodinger_sol_analytical_2D(x, y, t, a, b, kx0, ky0):
    f = 1/np.sqrt((1+ 2*1j*a*t)*(1+ 2*1j*b*t)) * (4*a*b/np.pi**2)**1/4 * np.exp(-(a*x**2 + b*y**2 - 1j*(kx0*x + ky0*y - t/2 * (kx0**2 + ky0**2) ))/((1+ 2*1j*a*t)*(1+ 2*1j*b*t)))
    return f

def tunnel(V0,d,x0,x1,dx):

    x = np.arange(x0,x1+dx,dx) 
    xn = np.shape(x)[0]

    V = np.zeros(xn)

    for i in range(xn):
        if np.abs(x[i]) < d/2:
            V[i] = V0
        else:
            continue
    return V 
a = 1
k_0 = 1
V0 = 1
d = 10


t0 = 0
t1 = 0.4
dt = 0.01

x0 = -20
x1 = 20
dx = 0.65

V=0

f1, X1, t = RK4(t0, t1, dt, x0, x1, dx, a, V, k_0=0)
prob_density_func1 = np.real(f1*np.conj(f1))
prob1 = sc.integrate.simpson(X1,prob_density_func1, dx=0.1)

# f2, X2, t = ftcs(t0, t1, dt, x0, x1, dx, a, V, k_0=k_0)
# prob_density_func2 = np.real(f2*np.conj(f2))
# prob2 = sc.integrate.simpson(X2,prob_density_func2, dx=0.1)

# f4, X4, t = crank_nichelson(t0, t1, dt, x0, x1, dx, a, k_0, V)
# prob_density_func4 = np.real(f4*np.conj(f4))
# prob4 = sc.integrate.simpson(X4,prob_density_func4, dx=0.1)

# f3 = schrodinger_sol_analytical(X4, t1, a, k_0)
# prob_density_func3 = np.real(f3*np.conj(f3))
# prob3 = sc.integrate.simpson(X4,prob_density_func3, dx=0.1)

print(prob1)

plt.figure()
# plt.plot(X2,prob_density_func2, label="FTCS")
# plt.plot(X4,prob_density_func4, label="Crank-Nichelson")
# plt.plot(X4,prob_density_func3, label="Analytic")
plt.plot(X1,prob_density_func1, label="RK4")
plt.ylim(0,1)
# plt.savefig("Group_test_RK4.png")
plt.legend()
plt.show()

# for i in range(len(t1)):
#     f, x, t = ftcs_1D(t0, t1[i], dt, x0, x1, dx, a, k_0=k_0)

#     prob_density_func = f*np.conj(f)

#     plt.figure()
#     plt.plot(x,prob_density_func)
#     #plt.savefig("Group_test_{0}.png".format([i]))
#     plt.show()