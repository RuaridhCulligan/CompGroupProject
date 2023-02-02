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
