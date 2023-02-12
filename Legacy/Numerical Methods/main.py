#-----------------------------------------------------------------------------
#
# "main.py" - FUNCTIONS TO IMPLEMENT AND VISUALISE NUMERICAL SOLUTIONS 
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

import os
from visualisation import visualise_1D, visualise_2D
from log_handling import create_log, read_log

def main(log_file="log.txt"):
    
    # if log file does not exist in location create log file with default values
    if os.path.exists(log_file)==False:
        create_log(log_file)
    
    # extract information from log files
    case, method, settings, sys_par, num_par=read_log(log_file)
    
    if case=="caseA" or case=="caseC" or case=="caseE":
        visualise_1D(case,method, settings, sys_par, num_par)
    elif case=="caseB" or case=="caseD":
        visualise_2D(case,method, settings, sys_par, num_par)                                         

# execute

#create_log("log.txt")

main()