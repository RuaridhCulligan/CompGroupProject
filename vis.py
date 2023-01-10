"""  
------------------------------------------------------------------------------------- 
    vis.py  - general style for visualisation outputs 
-------------------------------------------------------------------------------------
                                                        
"""     

# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# specify input output directories
in_dir  = ""
out_dir = ""
if os.path.exists(out_dir)==False:
    os.mkdir(out_dir) 

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