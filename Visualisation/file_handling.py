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

    # SHAPE OF CSV:
    # top row contains normalisation result + labels for each solution  ["commented out"]
    # second row contains x vals [horizontal, first col placeholder, last col placeholder]
    # all rows below: first col time value, other cols P(x,t), last col identifier in case there is several solutions





print(read_log("log.txt"))


