from data_sets.toy_data_var_complexity import toy_data_var_complexity
from data_sets.toy_data_var import toy_data_var

from Mondrian_Tree import Mondrian_Tree
from sklearn.tree import DecisionTreeRegressor

from examples.example_var_complexity_mt import example_var_complexity_mt

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy
import math

def  main():
    example_var_complexity_mt()


if __name__ == '__main__':
    main()