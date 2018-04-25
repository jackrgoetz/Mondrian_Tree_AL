import utils

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
        for _ in range(n_trees)
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