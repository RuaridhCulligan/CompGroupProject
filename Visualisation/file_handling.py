#-----------------------------------------------------------------------------
#
# "file_handling.py" - FUNCTIONS TO READ AND WRITE STANDARDISED FILES
#
#------------------------------------------------------------------------------
import numpy as np


# read in log files:
def read_log(path):
    arr = np.loadtxt(path,dtype="str", delimiter=" ", usecols=1)

    case = arr[0]
    method = arr[8]

    num_params = arr[1:8].astype("float")
    vis_params = arr[9:]


    return case, method, num_params, vis_params

# read in csv files:
def read_csv(path):

    #SHAPE of CSV:
    #   top row contains normalisation result
    #   2nd row contains x_vals [1st col: placeholder val]
    #   nth row contains P_vals [1st col: time val]

    t_vals = np.loadtxt(path, skiprows=2, usecols=0, delimiter=",")
    x_vals = np.loadtxt(path, skiprows=1, delimiter=",")[0,1:]
    P_vals = np.loadtxt(path, skiprows=2, delimiter=",")[:,1:]

    # read top row separately:
    with open(path, "r") as f:
        norm = f.readline()[:-1]
    f.close()

    #norm = norm.as_type("float")

    return t_vals, x_vals, P_vals, norm



print(read_csv("test.csv"))


