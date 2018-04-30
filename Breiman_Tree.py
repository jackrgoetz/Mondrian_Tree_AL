from sklearn.tree import DecisionTreeRegressor

import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

class Breiman_Tree:

    '''
    Main class for Breiman Tree version of active learning algorithm
    '''

    def __init__(self, seed=None):

        self.points = None
        self.labels = None
        self.labelled_index = None
        self._num_points = 0
        self._num_labelled = 0

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed