#-----------------------------------------------------------------------------
#
# "visualisation.py" - GENERAL PURPOSE VISUALISATION FUNCTIONS
#
#
#        Input: reads in data from csv file(s) in specified directory;
#               expects standardised file format
#
#        Output: produces visualisations as GIFs/PDFs in specified directory;
#                works with standardised file format
#
#        Dependencies: requires package 'celluloid' as well as other standard
#                      modules
#
#------------------------------------------------------------------------------

# import modules
import numpy as np
import celluloid
import matplotlib.pyplot as plt
import os
import pandas as pd

# relevant files and directories

in_dir        = "num_output"    # directory storing input files
in_filename   = "data.csv"      # standardise this
out_dir       = "vis_output"    # directory storing output files
out_filename  = "visualisation" # STANDARDISE THIS
out_filetype  = ".gif"          # filetype of output files

# set standardised layout of plots
fig_dim    = [8, 4]   # dimensions
title_size = 16       # title font size
body_size  = 14       # axes and legends font size
tick_size  = 12       # tick mark font size
plt.rcParams['text.usetex'] = True # enable LaTeX renadering
plt.rcParams['mathtext.fontset'] = 'cm' # use LateX font for maths
plt.rcParams['font.family'] = 'STIXGeneral' # use LateX font for text

# read log and data files

"""USE IMPORTED FUNCTIONS"""

# produce output



