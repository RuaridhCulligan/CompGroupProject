#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     AUXILIARY FUNCTIONS RELATING TO INITIAL CONDITIONS / ANALYTICAL SOLUTIONS

import numpy as np

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

def wavepacket_2particle(x, sys_params):
    a =  sys_params[1]
    k0 = sys_params[2]
    b   = sys_params[3]
    ky0 = sys_params[4]

    x0 = sys_params[9]
    y0 = sys_params[10]

    psi_2 = 1/np.sqrt(2) * (2*a/np.pi)**(1/4) * np.exp(-a*(x-x0)**2) * np.exp(1j*k0*(x-x0))
    psi_1 = 1/np.sqrt(2) * (2*b/np.pi)**(1/4) * np.exp(-b*(x-y0)**2) * np.exp(1j*ky0*(x-y0))
    
    return psi_1,psi_2
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