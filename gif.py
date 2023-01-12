#-----------------------------------------------------------------------------
#
# "gif.py" - visualises the analytical solution in the 1D case
#
#  NOTE: requires package 'celluloid' ; install via 'pip3 install celluloid'
#
#------------------------------------------------------------------------------

# import modules
from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
import os

# specify output file name and directory
out_dir = "output"
file_name = "1D_an_sol.gif"
if os.path.exists(out_dir)==False:
    os.mkdir(out_dir)
out_file = os.path.join(out_dir, file_name)    

# set parameter values
a = 1
k0 = 1

# define 1D analytical solution and its mod squared
def an_sol_1D(x, t, a, k0):
    return 1/(np.sqrt(2*np.pi))*((1/(2*a*np.pi))**(1/4))*np.exp(k0/4*a)*(1/np.sqrt((1/2*a)+1j*t))*(np.exp((-(x-((1j*k0)/(2*a)))**2)/((1/a)+2*1j*t)))

def mod2_sol(x,t,a,k0):
    return np.abs(an_sol_1D(x,t,a,k0))**2

def mod2_in_cond(x,a,k0):
    return np.abs((2*a/np.pi)**(0.25)*np.exp(-a*x*x+ 1j*k0*x))**2

# define ranges in space and time
x_min = -10
x_max = 10
Nx = 1000
t_min = 0
t_max = 10
Nt = 100
x = np.linspace(x_min, x_max, Nx)
t = np.linspace(t_min, t_max, Nt)

# set standardised layout of plots
fig_dim    = [8, 4]   # dimensions
file_type  = ".pdf"   # output file type
title_size = 16       # title font size
body_size  = 14       # axes and legends font size
tick_size  = 12       # tick mark font size 

# use Latex to render text and symbols
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

# set up plot
fig=plt.figure(figsize=fig_dim)
camera = Camera(fig)
for i in t:
    ax = plt.subplot(1,1,1)
    ax.text(x_min+0.3*(x_max-x_min),mod2_sol(x,0,a,k0).max(), 't={0:.3e}'.format(i), animated=True, fontsize=body_size, ha="center",va="bottom")
    ax.plot(x, mod2_sol(x,i,a,k0), color="red")
    ax.plot(x, mod2_in_cond(x,a,k0), color="gray", ls="--")
    ax.set_xlabel(r'x', fontsize=body_size)
    ax.set_ylabel(r'$|\Psi|^2$', fontsize=body_size)
    plt.title(r'Analytical Solution to the 1D Schr{\"o}dinger Equation', fontsize=title_size)
    ax.set_ylim(0, mod2_sol(x,0,a,k0).max()*1.1)
    plt.show()
    
    camera.snap()
    
# render and save animation
animation = camera.animate()
animation.save(out_file)
