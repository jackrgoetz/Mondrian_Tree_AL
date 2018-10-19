from data_sets.toy_data_var_complexity import toy_data_var_complexity
from data_sets.toy_data_var import toy_data_var

from Mondrian_Tree import Mondrian_Tree
from sklearn.tree import DecisionTreeRegressor

from examples.example_var_complexity_mt import example_var_complexity_mt
from examples.example_cl_mt import example_cl_mt
from examples.example_het_mt import example_het_mt
from examples.example_ccpp_mt import example_ccpp_mt
from examples.example_air_mt import example_air_mt
from examples.example_casp_mt import example_casp_mt
from examples.example_concrete_mt import example_concrete_mt
from examples.example_wine_mt import example_wine_mt

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy
import math

def  main():
    # example_var_complexity_mt()
    # example_het_mt()

    # example_cl_mt()
    # example_ccpp_mt()
    # example_air_mt()
    # example_casp_mt()
    # example_concrete_mt()
    example_wine_mt()


if __name__ == '__main__':
    main()