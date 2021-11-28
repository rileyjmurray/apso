This repository hosts code associated with the manuscript
"Algebraic perspectives on signomial optimization" (arXiv:2107.00345).
The code includes scripts in Python, Matlab, and Julia, as well
as a Python package called "apso".

## APSO

The apso Python package primarily implements our hierarchy of lower bounds
for (nonconvex) signomial minimization.
This package depends heavily on sageopt.
It also has functionality to convert a signomial optimization problem
represented in sageopt primitives into a pyomo model.
The pyomo model can be exported in GAMS format if you want to approach
a problem with a global nonlinear solver like BARON or SCIP.

I'll try to move apso code into sageopt eventually.
Until then, it's reasonable to install apso from source if you want
to try our methods.
Note that there isn't much source-code documentation, but the code
is simple, and you can see it in action with provided scripts.

## Scripts and text files

We have provided scripts to demonstrate our methods on three optimization
problems: a toy quadratic program in five variables (rm1978p23), a 
problem derived from chemical reaction network theory (crn), and a chemical
reactor design problem (bw1971).
These scripts require other software such as GAMS, Matlab, and Julia.
There are text files which we used to take notes or record solver output.

It's worth commenting on the files for the crn example.
 * crn0.py: this shows how we constructed the problem
  (namely, how we selected parameters that resulted in poor
  performance for the lowest-level SAGE relaxation).
 * crn1.py: this applies our paper's methods to the problem.
 * crn_matlab.m: this shows the results of approaching the problem
   with the moment-sos based "Lasserre hierarchy."
 * crn_julia.jl: this shows how to apply the CS-TSSOS Julia package
   to the example problem.
 * crn_julia_notes.txt: this has (very informal!) notes that I left
   to myself when exploring the CS-TSSOS parameter space in an
   interactive Julia session.
 
You will need to install MOSEK version 9.2 or higher in order to replicate our results.
