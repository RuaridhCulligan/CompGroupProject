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
from num_aux import integrate_1d, integrate_2d
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
file_name2D = "visualisation2D.pdf"

if os.path.exists(out_dir) ==False:
    os.mkdir(out_dir)

out_file  = os.path.join(out_dir, file_name)
out_file2D  = os.path.join(out_dir, file_name2D)

# visualise solution in one-dimensional case (cases A & C)
def visualise_1D(case,method, settings, sys_par, num_par):

    ADD_MET = settings[2]

    # make sure all input is sensible
    if method == ADD_MET:
            raise Exception("Additional method cannot be primary method.")
    if method == "an" and case != "caseA":
            raise Exception("No analytical solution is defined for this case.")               

    diff = settings[5]

    # compute relevant numerical solutions
    
    if method=="ftcs" and ADD_MET == "no":
        if case=="caseE":
            P, x, val, T = ftcs_2particle(case, settings, sys_par, num_par)
        else:    
            P, x, val, T = ftcs_1D(case, settings, sys_par, num_par)
    elif method=="rk4" and ADD_MET == "no":
        if case=="caseE":
            P, x, val, T = rk4_2particle(case, settings, sys_par, num_par)
        else:    
            P, x, val, T = rk4_1D(case, settings, sys_par, num_par)
    elif method=="cn" and ADD_MET == "no":
        if case=="caseE":
            P, x, val, T = cn_2particle(case, settings, sys_par, num_par)
        else:
            P, x, val, T = cn_1D(case, settings, sys_par, num_par)    
    elif method=="all" and ADD_MET == "no":
        if case=="caseE":
            P_ftcs, x, val_ftcs, T = ftcs_2particle(case, settings, sys_par, num_par)
            P_rk4, x, val_rk4, T   = rk4_2particle(case, settings, sys_par, num_par)
            P_cn, x, val_cn, T     = cn_2particle(case, settings, sys_par, num_par)
        else:    
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


        if float(settings[0]) == 0.0: 
            T   = np.arange(t_start, t_end, dt*100)

        if float(settings[0]) == 0.5:
            T   = np.array([t_start,t_end/8,t_end/4,t_end/2, 3*t_end/4, t_end])

        if float(settings[0]) == 1.0:
            T   = np.array([t_end])

        P = np.empty(len(T), dtype="object")
        for i in np.arange(len(T)):
            psi = an_sol_1D(x, T[i], sys_par)
            P[i] = np.abs(psi)**2

    
    # implement option to compute two numerical solutions using
    # the variable ADD_MET in the log file
    
    elif (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):
        if case=="caseE":
            P_ftcs, x, val_ftcs, T = ftcs_2particle(case, settings, sys_par, num_par)
            P_rk4, x, val_rk4, T = rk4_2particle(case, settings, sys_par, num_par)
        else:    
            P_ftcs, x, val_ftcs, T = ftcs_1D(case, settings, sys_par, num_par)
            P_rk4, x, val_rk4, T = rk4_1D(case, settings, sys_par, num_par)
    elif (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
        if case=="caseE":
            P_rk4, x, val_rk4, T = rk4_2particle(case, settings, sys_par, num_par)
            P_cn, x, val_cn, T = cn_2particle(case, settings, sys_par, num_par)
        else:    
            P_rk4, x, val_rk4, T = rk4_1D(case, settings, sys_par, num_par)
            P_cn, x, val_cn, T = cn_1D(case, settings, sys_par, num_par)
    elif (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
        if case=="caseE":
            P_ftcs, x, val_ftcs, T = ftcs_2particle(case, settings, sys_par, num_par)
            P_cn, x, val_cn, T = cn_2particle(case, settings, sys_par, num_par)
        else:    
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
        elif case=="caseE":
            plt.title(r'Collision of two Gaussian wavepackets with inter-particle potential (at $t={0:.3f})$'.format(sys_par[0]), fontsize=title_size)    
        

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
        plt.ylabel(r'Probability density $|\Psi(x,t)|^2$', fontsize=body_size)
        plt.xlabel(r'Spatial dimension $x$', fontsize=body_size)
        plt.savefig(out_file)
        plt.show()
 
    # produce visualisation in the non-static (GIF) case:
    if float(settings[0])==0:
        file_name_gif = "visualisation.gif"
        out_file_gif = os.path.join(out_dir, file_name_gif)
        fig = plt.figure(figsize=fig_dim)
        camera = Camera(fig)

        if case=="caseA":
            plt.title(r'Free propagation of a Gaussian wavepacket', fontsize=title_size)
        elif case=="caseC":
            plt.title(r'Tunneling of a Gaussian wavepacket', fontsize=title_size)
        if ADD_MET == "no":
            if method == "an":
                for i in np.arange(len(T)):
                    if case == "caseC":
                        raise Exception("There is no analytical solution for this case.")
                    ax = plt.subplot(1,1,1)
                    ax.text(45,P[0].max(),'t={0:.3e}'.format(T[i]), animated=True, fontsize=body_size, ha="center",va="bottom")
                    l = ax.plot(x,P[i],color="black", label= r'Analytical solution normalised to 1.0000')
                    ax.legend(l, [r"Analytical solution"],  loc="upper left")
                    ax.set_xlabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                    ax.set_ylabel(r'Spatial dimension $x$', fontsize=body_size)
                    camera.snap()

            elif method == "ftcs":
                if diff == "True" and case == "caseA":
                    P_diff = np.abs(P - P_an)
                    for i in np.arange(len(T)):
                        ax = plt.subplot(1,1,1)
                        ax.text(x.max(),P_diff[0].max(),'t={0:.3e}'.format(T[i]), animated=True, fontsize=body_size, ha="center",va="bottom")
                        l = ax.plot(x,P_diff[i],color="black")
                        if an==True and method != "an" and diff == "False":
                            ax.plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            ax.plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            ax.plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        ax.legend(l, [r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff[i],x))], loc="upper left", fontsize=body_size)
                        ax.set_xlabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        ax.set_ylabel(r'Spatial dimension $x$', fontsize=body_size)
                        camera.snap()
                else:
                    for i in np.arange(len(T)):
                        ax = plt.subplot(1,1,1)
                        ax.text(x.max(),P[0].max(),'t={0:.3e}'.format(T[i]), animated=True, fontsize=body_size, ha="center",va="bottom")
                        l = ax.plot(x,P[i],color="black")
                        if an==True and method != "an" and diff == "False":
                            ax.plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            ax.plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            ax.plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                            #ax.legend(p, [r'Finite potential well of depth {0:.3f}'.format(V0)], loc="upper left")
                        ax.legend(l, [r'FTCS scheme normalised to: {0:.3f}'.format(integrate_1d(P[i],x))], loc="upper left", fontsize=body_size)
                        ax.set_xlabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        ax.set_ylabel(r'Spatial dimension $x$', fontsize=body_size)
                        camera.snap()                  

        

        animation = camera.animate()
        animation.save(out_file_gif)
    
    # produce visualisation in the semi-static (subplot) case:
    if float(settings[0])==0.5:
        fig, axs = plt.subplots(3,2,figsize = [18,10])
        fig.subplots_adjust(hspace = .5, wspace=.4)

        if case=="caseA":
            plt.suptitle(r'Free propagation of a Gaussian wavepacket', fontsize=title_size)
        elif case=="caseC":
            plt.suptitle(r'Tunneling of a Gaussian wavepacket', fontsize=title_size)
        elif case=="caseE":
            plt.suptitle(r'Collision of two Gaussian wavepackets with inter-particle potential (at $t={0:.3f})$'.format(sys_par[0]), fontsize=title_size)


        if ADD_MET == "no":
            axs = axs.ravel()
            if method=="ftcs":
                if diff == "True" and case == "caseA":
                    P_diff = np.abs(P - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff[i],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff[i],x)))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P[i],color="black", label= r'FTCS scheme normalised to {0:.4f}'.format(val[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)

            elif method=="rk4": 
                if diff == "True" and case == "caseA":
                    P_diff = np.abs(P - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff[i],color="black", label=r'Error on RK4 method (total: {0:.3f})'.format(integrate_1d(P_diff[i],x)))
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--") 
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P[i],color="black", label= r'RK4 method normalised to {0:.4f}'.format(val[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")  
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)


            elif method=="cn": 
                if diff == "True" and case == "caseA":
                    P_diff = np.abs(P - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff[i],color="black", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff[i],x)))
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--") 
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P[i],color="black", label= r'CN scheme normalised to {0:.4f}'.format(val[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution') 
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)

            
            elif method=="all":
                if diff == "True" and case == "caseA":
                    P_diff1 = np.abs(P_ftcs - P_an)
                    P_diff2 = np.abs(P_rk4 - P_an)
                    P_diff3 = np.abs(P_cn - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff1[i],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff1[i],x)))
                        axs[i].plot(x,P_diff2[i],color="gray", label=r'Error on RK4 method (total: {0:.3f})'.format(integrate_1d(P_diff2[i],x)))
                        axs[i].plot(x,P_diff3[i],color="blue", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff3[i],x)))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--") 
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_ftcs[i],color="black", label= r'FTCS scheme normalised to {0:.4f}'.format(val[i]))
                        axs[i].plot(x,P_rk4[i],color="gray", label= r'RK4 scheme normalised to {0:.4f}'.format(val[i]))
                        axs[i].plot(x,P_cn[i],color="blue", label= r'CN scheme normalised to {0:.4f}'.format(val[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')       
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")                   
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)


            elif method == "an":
                for i in range(len(T)):
                    axs[i].set_title("t={0:.3f}".format(T[i]))
                    axs[i].plot(x,P[i],color="black", label= r'Analytical solution normalised to {0:.4f}'.format(integrate_1d(P[i],x)))                       
                    axs[i].legend(fontsize=body_size, loc="upper right")
                    axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                    axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)

        else:
            axs = axs.ravel()
            if (method == "rk4" and ADD_MET == "ftcs") or (method == "ftcs" and ADD_MET == "rk4"):
                if diff == "True" and case == "caseA":
                    P_diff1 = np.abs(P_rk4 - P_an)
                    P_diff2 = np.abs(P_ftcs - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff1[i],color="black", label=r'Error on RK4 method (total: {0:.3f})'.format(integrate_1d(P_diff1[i],x)))
                        axs[i].plot(x,P_diff2[i],color="gray", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff2[i],x)))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')  
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_rk4[i],color="black", label= r'RK4 method normalised to {0:.4f}'.format(val_rk4[i]))
                        axs[i].plot(x,P_ftcs[i],color="gray", label= r'FTCS scheme normalised to {0:.4f}'.format(val_ftcs[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution') 
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")                      
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)


            if (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
                if diff == "True" and case == "caseA":
                    P_diff1 = np.abs(P_rk4 - P_an)
                    P_diff2 = np.abs(P_cn - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff1[i],color="black", label=r'Error on RK4 method (total: {0:.3f})'.format(integrate_1d(P_diff1[i],x)))
                        axs[i].plot(x,P_diff2[i],color="gray", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff2[i],x)))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')  
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_rk4[i],color="black", label= r'RK4 method normalised to {0:.4f}'.format(val_rk4[i]))
                        axs[i].plot(x,P_cn[i],color="gray", label= r'CN scheme normalised to {0:.4f}'.format(val_cn[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution') 
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")                   
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)


            elif (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
                if diff == "True" and case == "caseA":
                    P_diff1 = np.abs(P_ftcs - P_an)
                    P_diff2 = np.abs(P_cn - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_diff1[i],color="black", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_1d(P_diff1[i],x)))
                        axs[i].plot(x,P_diff2[i],color="gray", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_1d(P_diff2[i],x)))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')  
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot(x,P_ftcs[i],color="black", label= r'FTCS scheme normalised to {0:.4f}'.format(val_ftcs[i]))
                        axs[i].plot(x,P_cn[i],color="gray", label= r'CN scheme normalised to {0:.4f}'.format(val_cn[i]))
                        if an==True and method != "an" and diff == "False":
                            axs[i].plot(x,P_an[i],color="red",linestyle="--", label=r'Analytical solution')
                        if v == True:
                            axs[i].plot(V_x,np.array([0,P[0].max()]),color="green",linestyle="--", label=r'Finite potential well of depth {0:.4f} '.format(V0))
                            axs[i].plot(-V_x,np.array([0,P[0].max()]),color="green",linestyle="--")                 
                        axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
    
        
        plt.legend(fontsize=body_size, loc="upper right")
        plt.savefig(out_file)
        plt.show()
    
    return 0


# visualise solution in two-dimensional case (cases B & D)
def visualise_2D(case,method, settings, sys_par, num_par):
    ADD_MET = settings[2]

    # make sure all input is sensible
    if method == ADD_MET:
            raise Exception("Additional method cannot be primary method.")
    if method == "an" and case != "caseB":
            raise Exception("No analytical solution is defined for this case.")               

    diff = settings[5]

    # compute relevant numerical solutions
    
    if method=="ftcs" and ADD_MET == "no":    
        P, x,y, val, T = ftcs_2D(case, settings, sys_par, num_par)
    elif method=="rk4" and ADD_MET == "no":   
        P, x,y, val, T = rk4_2D(case, settings, sys_par, num_par)
    elif method=="cn" and ADD_MET == "no":
        P, x, val, T = cn_2D(case, settings, sys_par, num_par)    
    elif method=="all" and ADD_MET == "no":
        raise Exception("Overlay of different solutions can not be visualised in 2D.")
    elif method=="an":
        x_min   = num_par[0]
        x_max   = num_par[1]
        dx      = num_par[2]
        t_start = 0
        t_end   = sys_par[0]
        dt      = num_par[3]
        y_min   = num_par[4]
        y_max   = num_par[5]
        dy      = num_par[6]
        x = np.arange(x_min, x_max+dx, dx)
        y = np.arange(y_min, y_max+dy, dy)
        X, Y = np.meshgrid(x,y)


        if float(settings[0]) == 0.0: 
            T   = np.arange(t_start, t_end, dt*100)

        if float(settings[0]) == 0.5:
            T   = np.array([t_start,t_end/8,t_end/4,t_end/2, 3*t_end/4, t_end])

        if float(settings[0]) == 1.0:
            T   = np.array([t_end])

        P = np.empty(len(T), dtype="object")
        for i in np.arange(len(T)):
            psi = an_sol_2D(X,Y, T[i], sys_par)
            P[i] = np.abs(psi)**2

    
    
    # implement option to compute two numerical solutions using
    # the variable ADD_MET in the log file
    
    elif (method == "ftcs" and ADD_MET == "rk4") or (method == "rk4" and ADD_MET == "ftcs"):   
        raise Exception("Overlay of different solutions can not be visualised in 2D.")
    elif (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
        raise Exception("Overlay of different solutions can not be visualised in 2D.")
    elif (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
        raise Exception("Overlay of different solutions can not be visualised in 2D.")
    
    # define spatial grid
    X, Y = np.meshgrid(x,y)

    # implement option to display potential in relevant cases
    SHOW_V = float(settings[3])
    v = False
    V0 = sys_par[5]
    
    if SHOW_V==1 and case=="caseB":
            d = sys_par[6]
            w = sys_par[7]
            v = True
            
    # compute analytical solution in relevant cases:
    an = False
    
    if (case=="caseB" and float(settings[1])!=1) or diff == "True":
        an = True
        
        if method != "all" and ADD_MET == "no":
            P_an = np.copy(P)
                 
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
        
        if method=="ftcs" and ADD_MET == "no":
            if diff == "True" and case == "caseB":
                P_diff = np.abs(P - P_an)
                ax.plot_surface(X,Y,P_diff[0],color="black", cmap="binary", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_2d(P_diff[0],x,y)))

            else:  
                ax.plot_surface(X,Y,P[0], color="black" ,cmap="binary", label=r'FTCS scheme normalised to {0:.4f} '.format(val[0]))    

            if v == True:
                ax.set_xlim(-d/2, d/2)
                ax.set_ylim(-w/2, w/2)

        elif method=="rk4" and ADD_MET == "no": 
            if diff == "True" and case == "caseB":
                P_diff = np.abs(P - P_an)
                ax.plot_surface(X,Y,P_diff[0],color="black", cmap="binary", label=r'Error on RK4 scheme (total: {0:.3f})'.format(integrate_2d(P_diff[0],x,y)))
            else:
                ax.plot_surface(X,Y,color="black", cmap="binary", label=r'RK4 method normalised to {0:.4f} '.format(val[0]))       

            if v == True:
                ax.set_xlim(-d/2, d/2)
                ax.set_ylim(-w/2, w/2)
       
        elif method=="cn" and ADD_MET == "no": 
            if diff == "True" and case == "caseB":
                P_diff = np.abs(P - P_an)
                ax.plot_surface(X,Y,P_diff[0],color="black", cmap="binary", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_2d(P_diff[0],x,y)))
            else:
                ax.plot_surface(X,Y,P[0],color="black", cmap="binary", label=r'CN scheme normalised to {0:.4f} '.format(val[0]))

            if v == True:
                ax.set_xlim(-d/2, d/2)
                ax.set_ylim(-w/2, w/2)
        
        elif method=="all" and ADD_MET == "no":
            raise Exception("Overlay of different solutions can not be visualised in 2D.")

        elif (method=="an" and ADD_MET=="no" and case=="caseB"):
            ax.plot_surface(X,Y,P[0],color="black", cmap="binary", label=r'Analytical solution normalised to {0:.4f}'.format(integrate_2d(P[0],x,y)))

            if v == True:
                ax.set_xlim(-d/2, d/2)
                ax.set_ylim(-w/2, w/2)
        
        #ax.legend(fontsize=body_size, loc="upper right")
        ax.set_zlabel(r'Probability density $|\Psi(x,y,t)|^2$', fontsize=body_size)
        ax.set_xlabel(r'Spatial dimension $x$', fontsize=body_size)
        ax.set_ylabel(r'Spatial dimension $y$', fontsize=body_size)
        plt.savefig(out_file2D)
        plt.show()

    # produce visualisation in the non-static (GIF) case:
        
        """
        to do! 
        """    
    
    # produce visualisation in the semi-static (subplot) case:
    if float(settings[0])==0.5:
        fig, axs = plt.subplots(2,3,figsize = [18,10], subplot_kw={'projection': '3d'})
        fig.subplots_adjust(hspace = .2, wspace=.1) # h .5 w .4

        if case=="caseB":
            fig.suptitle(r'Free propagation of a Gaussian wavepacket'.format(sys_par[0]), fontsize=title_size)
        elif case=="caseD":
            fig.suptitle(r'Gaussian wavepacket in an infinite potential well '.format(sys_par[0]), fontsize=title_size)


        if ADD_MET == "no":
            axs = axs.ravel()
            if method=="ftcs":
                if diff == "True" and case == "caseB":
                    P_diff = np.abs(P - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot_surface(X,Y,P_diff[i],color="black", cmap="binary", label=r'Error on FTCS scheme (total: {0:.3f})'.format(integrate_2d(P_diff[i],x,y)))
                        
                        if v == True:
                            axs[i].set_xlim(-d/2, d/2)
                            axs[i].set_ylim(-w/2, w/2)
                        #axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'$x$', fontsize=body_size)
                        axs[i].set_ylabel(r'$y$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot_surface(X,Y,P[i],color="black", cmap="binary", label= r'FTCS scheme normalised to {0:.4f}'.format(val[i]))
                        
                        if v == True:
                            axs[i].set_xlim(-d/2, d/2)
                            axs[i].set_ylim(-w/2, w/2)
                        #axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'$x$', fontsize=body_size)
                        axs[i].set_ylabel(r'$y$', fontsize=body_size)

            elif method=="rk4": 
                if diff == "True" and case == "caseB":
                    P_diff = np.abs(P - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot_surface(X,Y,P_diff[i],color="black", cmap="binary", label=r'Error on RK4 method (total: {0:.3f})'.format(integrate_2d(P_diff[i],x,y)))
                        if v == True:
                            axs[i].set_xlim(-d/2, d/2)
                            axs[i].set_ylim(-w/2, w/2)
                        #axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'$x$', fontsize=body_size)
                        axs[i].set_ylabel(r'$y$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot_surface(X,Y,P[i],color="black", cmap="binary", label= r'RK4 method normalised to {0:.4f}'.format(val[i]))
                        
                        if v == True:
                            axs[i].set_xlim(-d/2, d/2)
                            axs[i].set_ylim(-w/2, w/2)
                        #axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'$x$', fontsize=body_size)
                        axs[i].set_ylabel(r'$y$', fontsize=body_size)


            elif method=="cn": 
                if diff == "True" and case == "caseB":
                    P_diff = np.abs(P - P_an)
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot_surface(X,Y,P_diff[i],color="black", cmap="binary", label=r'Error on CN scheme (total: {0:.3f})'.format(integrate_2d(P_diff[i],x,y)))
                        if v == True:
                            axs[i].set_xlim(-d/2, d/2)
                            axs[i].set_ylim(-w/2, w/2)
                        #axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'$x$', fontsize=body_size)
                        axs[i].set_ylabel(r'$y$', fontsize=body_size)
                else:
                    for i in range(len(T)):
                        axs[i].set_title("t={0:.3f}".format(T[i]))
                        axs[i].plot_surface(X,Y,P[i],color="black", cmap="binary", label= r'CN scheme normalised to {0:.4f}'.format(val[i]))
                        
                        if v == True:
                            axs[i].set_xlim(-d/2, d/2)
                            axs[i].set_ylim(-w/2, w/2)

                        #axs[i].legend(fontsize=body_size, loc="upper right")
                        axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                        axs[i].set_xlabel(r'$x$', fontsize=body_size)
                        axs[i].set_ylabel(r'$y$', fontsize=body_size)

            
            elif method=="all":
                raise Exception("Overlay of different solutions can not be visualised in 2D.")


            elif method == "an":
                for i in range(len(T)):
                    axs[i].set_title("t={0:.3f}".format(T[i]))
                    axs[i].plot_surface(X,Y,P[i],color="black", cmap="binary", label= r'Analytical solution normalised to {0:.4f}'.format(integrate_2d(P[i],x,y)))                       
                    #axs[i].legend(fontsize=body_size, loc="upper right")
                    axs[i].set_zlabel(r'$|\Psi(x,y,t)|^2$', fontsize=body_size)
                    axs[i].set_xlabel(r'$x$', fontsize=body_size)
                    axs[i].set_ylabel(r'$y$', fontsize=body_size)


        else:
            if (method == "rk4" and ADD_MET == "ftcs") or (method == "ftcs" and ADD_MET == "rk4"):
                raise Exception("Overlay of different solutions can not be visualised in 2D.")
            elif (method == "rk4" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "rk4"):
                raise Exception("Overlay of different solutions can not be visualised in 2D.")
            elif (method == "ftcs" and ADD_MET == "cn") or (method == "cn" and ADD_MET == "ftcs"):
                raise Exception("Overlay of different solutions can not be visualised in 2D.")
            
        plt.savefig(out_file2D)
        plt.show()
      

    return 0

