import random
import copy
import utils
import warnings
import math

from LeafNode import LeafNode
from SplitNode import SplitNode
from Mondrian_Tree import Mondrian_Tree

class Mondrian_Forest:

    '''
    Main class for Mondrian Forest, a forest of Mondrian Trees
    '''

    def __init__(self, linear_dims, n_trees):
        self._linear_dims = linear_dims
        self._num_dimensions = len(linear_dims)
        self._n_trees = n_trees
        self.tree_list = []
        for _ in range(n_trees):
            self.tree_list.append(Mondrian_Tree(self._linear_dims))

        self.points = None
        self.labels = None
        self._num_points = 0
        self._num_labelled = 0

        self._life_time = 0

    def __str__(self):
        # Add more as needed
        return (
        'Number of dimensions = {}\n' 
        'Number of trees = {}\n'
        'Life time parameter = {}\n'
        '\n'
        'Number of data points = {}\n'
        'Number of labels = {}'.format(
            self._num_dimensions, self._life_time, self._n_trees, self._num_points, 
            self._num_labelled))

    def _test_point(self, new_point):
        '''Tests an input point, raising errors if it's a bad type and converting it from
        a numpy array to a list if needed 
        '''

        try:
            len(new_point)
        except TypeError:
            raise TypeError(
                'Given point has no len(), so probably is not a vector representing a data point. '
                'Try turning it into a list, tuple or numpy array where each entry is a dimension.')

        if len(new_point) != self._num_dimensions:
            raise ValueError(
                'Data point is not of the correct length. Must be the same dimension as the '
                'dimensions used to build the Mondrian Tree when it was initialized.')

        if str(type(new_point)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting new_point to list internally')
            new_point = new_point.tolist()

        if type(new_point) not in [list, tuple]:
            raise TypeError('Please input the new point as a list, tuple or numpy array')

        return new_point

    ###########################################

    def update_life_time(self, new_life_time, set_seeds = None):

        for i, tree in enumerate(self.tree_list):
            if set_seeds is not None:
                tree.update_life_time(new_life_time, set_seeds[i])
            else:
                tree.update_life_time(new_life_time)

    def input_data(self, all_data, labelled_indicies, labels):

        all_data = copy.deepcopy(all_data)
        labelled_indicies = copy.deepcopy(labelled_indicies)
        labels = copy.deepcopy(labels)

        if len(all_data) < len(labelled_indicies):
            raise ValueError('Cannot have more labelled indicies than points')

        if len(labelled_indicies) != len(labels):
            raise ValueError('Labelled indicies list and labels list must be same length')

        for point in all_data:
            if len(point) != self._num_dimensions:
                raise ValueError('All data points must be of the dimension on which this \
                    Mondrian Tree is built ({})'.format(self._num_dimensions))

        if str(type(all_data)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting all_data to list of lists internally')
            all_data = all_data.tolist()

        if str(type(labelled_indicies)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labelled_indicies to list internally')
            labelled_indicies = labelled_indicies.tolist()

        if str(type(labels)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labels to list internally')
            labels = labels.tolist()

        self.points = all_data
        self._num_points = len(self.points)
        self._num_labelled = len(labels)

        # Making a label list, with None in places where we don't have the label

        temp = [None] * self._num_points
        for i,ind in enumerate(labelled_indicies):
            temp[ind] = labels[i]
        self.labels = temp

        for i, tree in enumerate(self.tree_list):
            tree.input_data(all_data, labelled_indicies, labels)
            tree.points = self.points
            tree.labels = self.labels
            
    def label_point(self, index, value):
        self._num_labelled += 1
        for tree in self.tree_list:
            tree.label_point(index, value)

    def add_data_point(self, new_point, label = None):

        new_point = copy.deepcopy(new_point)
        label = copy.deepcopy(label)
        new_point = self._test_point(new_point)

        if self.points is None:
            point_index = 0
            self.points = [new_point]
            self.labels = [label]
        else:
            point_index = len(self.labels)
            self.points.append(new_point)
            self.labels.append(label)

        self._num_points += 1
        if label is not None:
            self._num_labelled += 1

        for tree in self.tree_list:
            leaf = tree._root.leaf_for_point(new_point)
            if label is None:
                leaf.unlabelled_index.append(point_index)
                tree._num_points += 1
            else:
                leaf.labelled_index.append(point_index)
                tree._num_points += 1
                tree._num_labelled += 1

            tree._full_leaf_marginal_list_up_to_date = False
            if label is not None:
                tree._full_leaf_mean_list_up_to_date = False
                tree._full_leaf_var_list_up_to_date = False

