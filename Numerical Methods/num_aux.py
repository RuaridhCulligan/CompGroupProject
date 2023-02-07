#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     AUXILIARY NUMERICAL SOLVERS

import numpy as np

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
def tridiag_solver(a, b, c, d):
    nf = len(d) # number of equations
    ac, bc, cc, dc = (x.astype(float) for x in (a, b, c, d)) # copy arrays & cast to floats
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]
 
    xc = bc
    xc[-1] = dc[-1]/bc[-1]
 
    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
 
    return xc
