from data_sets.toy_data_var_complexity import toy_data_var_complexity
from data_sets.toy_data_var import toy_data_var

from Mondrian_Tree import Mondrian_Tree
from sklearn.tree import DecisionTreeRegressor

from examples.example_var_mt import example_var_mt
from examples.example_het_mt import example_het_mt
from examples.example_cl_mt import example_cl_mt
from examples.example_ccpp_mt import example_ccpp_mt
from examples.example_air_mt import example_air_mt
from examples.example_casp_mt import example_casp_mt
from examples.example_concrete_mt import example_concrete_mt
from examples.example_wine_mt import example_wine_mt

from Mondrian_Forest import Mondrian_Forest
from sklearn.ensemble import RandomForestRegressor

from examples.example_var_mf import example_var_mf
from examples.example_het_mf import example_het_mf
from examples.example_ccpp_mf import example_ccpp_mf
from examples.example_cl_mf import example_cl_mf
from examples.example_air_mf import example_air_mf
from examples.example_casp_mf import example_casp_mf
from examples.example_concrete_mf import example_concrete_mf
from examples.example_wine_mf import example_wine_mf

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy
import math

def  main():
    '''
    Graphs and resuls will be saved in the graphs directory. 
    To change parameters of the experiemnts change them in the
    respective python scripts. 
    '''

    # Mondrian Tree examples

    # example_var_mt()
    # example_het_mt()

    # example_cl_mt()
    # example_ccpp_mt()
    # example_air_mt()
    # example_casp_mt()
    # example_concrete_mt()
    # example_wine_mt()

    # Mondrian Forest examples 

    # example_var_mf()
    # example_het_mf()

    # example_ccpp_mf()
    # example_cl_mf()
    # example_air_mf()
    # example_casp_mf()
    # example_concrete_mf()
    example_wine_mf()


if __name__ == '__main__':
    main()