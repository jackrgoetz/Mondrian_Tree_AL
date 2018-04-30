#!/bin/bash

# module purge
module load python-anaconda3/latest

cd Mondrian_Tree
qsub runMondrian_Forest_ccpp.sh
qsub runMondrian_Forest_cl.sh
# qsub runMondrian_Forest_var.sh
# qsub runMondrian_Forest_var_complexity.sh
