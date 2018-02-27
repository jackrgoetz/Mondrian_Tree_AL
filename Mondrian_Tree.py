import random
from LeafNode import LeafNode
from SplitNode import SplitNode

class Mondrian_Tree:

    def __init__(self, linear_dims):
        self._max_linear_dims = linear_dims
        self._root = LeafNode(linear_dims = self._max_linear_dims)
        self._num_dimensions = len(linear_dims)

        self.covariates = None
        self.labels = None
        self._num_points = 0

        self._life_time = 0
        self._num_leaves = 0

    def __str__(self):
        # Add more as needed
        return '\
        Number of dimensions = {}\n \
        Number of leaf nodes = {}\n\
        Life time parameter = {}\n\
        \n\
        Number of data points = {}'.format(
            self._num_dimensions, self._num_leaves,
            self._life_time, self._num_points)

    def update_life_time(self, new_life_time):

        '''Function for updating the tree with a new life time parameter, potentially 
        growing the tree.
        '''

        old_life_time = self._life_time
        self._life_time = new_life_time

        
