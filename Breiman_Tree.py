from sklearn.tree import DecisionTreeRegressor
from collections import Counter

import utils
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy

class Breiman_Tree:

    '''
    Main class for Breiman Tree version of active learning algorithm
    '''

    def __init__(self, min_samples_leaf=None, seed=None):

        self.points = None
        self.labels = None
        self.labelled_indices = None
        self._num_points = 0
        self._num_labelled = 0

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        if min_samples_leaf is None:
            self.min_samples_leaf=1
        else:
            self.min_samples_leaf=min_samples_leaf

        self.tree = DecisionTreeRegressor(random_state=self.seed,min_samples_leaf=self.min_samples_leaf)
        self._leaf_indices = []
        self._leaf_marginal = []
        self._leaf_var = []
        self._al_proportions =[]

        self._leaf_statistics_up_to_date = False

        self._verbose = False

    def input_data(self, all_data, labelled_indices, labels, copy_data=True):

        if copy_data:
            all_data = copy.deepcopy(all_data)
            labelled_indices = copy.deepcopy(labelled_indices)
            labels = copy.deepcopy(labels)

        if len(all_data) < len(labelled_indices):
            raise ValueError('Cannot have more labelled indicies than points')

        if len(labelled_indices) != len(labels):
            raise ValueError('Labelled indicies list and labels list must be same length')

        if str(type(all_data)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting all_data to list of lists internally')
            all_data = all_data.tolist()

        if str(type(labelled_indices)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labelled_indices to list internally')
            labelled_indices = labelled_indices.tolist()

        if str(type(labels)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labels to list internally')
            labels = labels.tolist()

        self.points = all_data
        self._num_points = len(self.points)
        self._num_labelled = len(labels)

        # Making a label list, with None in places where we don't have the label

        temp = [None] * self._num_points
        for i,ind in enumerate(labelled_indices):
            temp[ind] = labels[i]
        self.labels = temp
        self.labelled_indices = list(labelled_indices)

    def fit_tree(self):
        self.tree.fit(np.array(self.points)[self.labelled_indices,:], 
            np.array(self.labels)[self.labelled_indices])
        self._leaf_indices = self.tree.apply(np.array(self.points))
        self._leaf_statistics_up_to_date = False

    def label_point(self, index, value):

        if self.labels is None:
            raise RuntimeError('No data in the tree')

        if len(self.labels) <= index:
            raise ValueError('Index {} larger than size of data in tree'.format(index))

        value = copy.copy(value)
        index = copy.copy(index)

        self.labels[index] = value
        self.labelled_indices.append(index)

    def calculate_leaf_statistics(self):
        temp = Counter(self._leaf_indices)
        self._leaf_marginal = []
        self._leaf_var = []
        for key in np.unique(self._leaf_indices):
            self._leaf_marginal.append(temp[key]/self._num_points)
            temp_ind = [i for i,x in enumerate(self._leaf_indices) if x == key]
            temp_labels = [x for x in self.labels if x is not None]
            self._leaf_var.append(utils.unbiased_var(temp_labels))

    def al_calculate_leaf_proportions(self):
