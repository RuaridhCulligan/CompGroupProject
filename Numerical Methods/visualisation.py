#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     VISUALISATION FUNCTIONS

# import modules
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import os

from ftcs import ftcs_1D, ftcs_2D, ftcs_2particle
from rk4 import rk4_1D, rk4_2D, rk4_2particle
from cn import cn_1D, cn_2D, cn_2particle
from num_aux import integrate_1d, integrate_1d
from wavefuncs import an_sol_1D, an_sol_2D

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
file_name = "visualisation.pdf"

if os.path.exists(out_dir) ==False:
    os.mkdir(out_dir)

out_file  = os.path.join(out_dir, file_name)


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
        plt.savefig(out_file)
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
        animation.save(out_file)
    
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
        plt.savefig(out_file)
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