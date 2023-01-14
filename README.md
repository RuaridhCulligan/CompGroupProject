# P3 Comp Group Project: Dynamics of a Wavepacket
*Dillon Leahy, David Amorim, Ruaridh Culligan*

## General Overview

**Scope**
Numerically solve the Schrodinger equation for the following cases:
        A) Free wavepacket in 1D
        B) Free wavepacket in 2D
        C) Wavepacket in a 1D finite potential well
        D) Wavepacket in a 2D infinite potential well
        E) Collision between two wavepackets
        F) ... (potentially add more) ...

In cases A) and B) the accuracy of the numerical solution can be determined
by direct comparison with the analytical solution. In cases C) through F) this is
not possible. The metric of evaluation for the latter cases is the normalisation
condition on the wavefunctions which can be tested using numerical integration.

*Note: the cases will internally be referred to using their alphabetical labels and*
*the labels should therefore not be used in other contexts.*


**Product**
The aim is to create a software package with the following front-end user flow:

        1) the user starts the program, resulting in a GUI being loaded
        2) the user selects a case to evaluate
        3) the user specifies relevant parameters and "submits"
        4) the GUI presents a visualisation of the corresponding wavefunction
           containing all relevant information
        5) the user has the option to save this output
        6) the user has the option to close the program or return to 2)

The corresponding back-end architecture is discussed below.

**Report**
TBD

**Presentation**
TBD

## Administrative Details

**Individual Responsibilities**
While all group members are involved in all parts of the projects, the following specialisations apply:
 - Ruaridh: GUI, GitHub
 - David: Visualisation, Latex, "secretary"
 - Dillon: Numerical methods

**Meetings**
Supervisor meetings on Tuesays during lab time. Group meetings on Tuesdays and Thursdays during lab time.
Additional meetings whenever necessary.

**Record Keeping**
Key points discussed in meetings will be added to this document by David.


## Code Structure

The code is partitioned into three independent blocks that communicate by reading and writing standardised files. The general structure is:

                        GUI ----(.log)----> NUM ----(.csv)----> VIS ----(.gif)----> GUI

which can be interpeted as the flattened representation of a circular relationship. The three blocks are the following:

**GUI**
The *GUI* block consists of a combination of Python and HTML files and handles all user front-end interaction in line with the user flow described above. The *GUI* block writes all collected user input into a *.log* file with standardised format. It reads in *.gif* files from a standardised location and displays them accordingly.

**NUM**
The *NUM* block consists of one (or several) Python file(s). It reads the *.log* file produced by the *GUI* block and generates numerical output accordingly. This output is then written to a *.csv* file of standard format.

**VIS**
The *VIS* block consists of one Python file. It reads the *.csv* file produced by the *NUM* block and produces an appropriate visualisation that is then saved as a *.gif*.


## File Conventions

For the independet blocks *GUI*, *NUM* and *VIS* to work together there must be established conventions for the three types of auxiliary and output files produced (i.e. *.log*, *.csv*, *.gif*). These file conventions concern the name, content and location of the relevant files.

**Relevant Content**
The following content must be contained in each of the three file types:
        1)*.log* :
                - case
                - start time, end time
                - spatial grid values
                - parameters of initial condition/boundaries
                - toggle: animated/static/semistatic output
                - integration method
                - toggle: overlay other solutions
                - ... (?)
        2) *.csv* :
                - mod squared of wavefunction (P) values
                - associated times and positions
                - result of normalisation test (with error?)
                - ... (add the remaining *.log* info?)
        3) *.gif* :
                - simply a static or animated image file
                - (requires info from *.log* or *.csv* for appropriate labelling)

**Convention for *.log* Files**
TBD.

*Suggestion:* lines of format "VARIABLE = value" with specified variable names and orders. Where certain variables are not applicable, simply leave blank: "VARIABLE = ". Naming convention: simply "log.log". Directory: wherever convenient for GUI.

**Convention for *.csv* Files**
TBD.

*Suggestion:* columns 'P', 'x', ('y'), 't', ('sol') where 'P' is the mod squared wavefunction at a given point, 'x', 't' (and if relevant 'y') associated point, 'sol' the nature of the solution (i.e. analytical vs numerical or different methods of numerical) where appropriate ; normalisation results and other relevant input from the *.log* file as header comments. Naming convention: "data.csv". Directory: wherever convenient for GUI.

**Convention for *.gif* Files**
TBD.

*Suggestion:* name "vis.gif" and save in location convenient for GUI.

## Outline

**Week 1 (09/01 - 15/01):**
- fix analytical solution
- decide on file conventions

**Near Future:**
- research and experiment with numerical methods for SE (including stability and error)
- research and experiment with numerical integration methods for normalisation (including error)
- work on individual code blocks and input/output standardisation
*Aim: produce a working protype for the simplest case (1D free wavepacket)*

**Middle Future**
- generalise to the other cases








