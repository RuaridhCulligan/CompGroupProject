#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     SYSTEM FUNCTIONS RELATING TO WRITING / READING LOG FILES

import numpy as np

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