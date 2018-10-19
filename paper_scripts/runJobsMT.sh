#!/bin/bash

# module purge
module load python-anaconda3/latest

cd Mondrian_Tree
qsub runMondrian_Tree_ccpp.sh
qsub runMondrian_Tree_cl.sh
qsub runMondrian_Tree_var.sh
qsub runMondrian_Tree_var_complexity.sh
