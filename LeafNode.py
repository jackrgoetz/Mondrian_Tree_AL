import random
import math
import copy

class LeafNode:

    '''Nodes for tree leaves

    Args:
        labelled_index (list): Index for labelled data points in the leaf.
        unlabelled_index (list): Index for unlabelled data points in the leaf.
        linear_dims (list): A p dim list of 2 dim lists indicating the upper and 
        lower bounds of the leaf.
        parent_node (SplitNode): This node's parent
        parent_branch (int): This node's branch (0 for left, 1 for right)
    '''

    # leaf_ids = 0

    def __init__(self, labelled_index=None, unlabelled_index=None,linear_dims=None,
        parent_node=None, parent_branch=None):
        self.labelled_index = copy.deepcopy(labelled_index)
        if self.labelled_index is None:
            self.labelled_index = []
        self.unlabelled_index = copy.deepcopy(unlabelled_index)
        if self.unlabelled_index is None:
            self.unlabelled_index = []
        self.linear_dims = copy.deepcopy(linear_dims)
        if self.linear_dims is None:
            self.linear_dims = []
        self.parent_node = parent_node
        self.parent_branch = parent_branch
        self.subtree_linear_dim = self.calculate_subtree_linear_dim()

        # LeafNode.leaf_ids+= 1
        # self.leaf_id = LeafNode.leaf_ids
        self.leaf_id = None

        self.full_leaf_list_pos = None

    def __str__(self):
        print_str = (
        'n_labelled = {}\n'
        'n_unlabelled = {}\n'
        'leaf_id = {}'.format(
            len(self.labelled_index), 
            len(self.unlabelled_index),
            self.leaf_id))
        return(print_str)

    def rounded_linear_dims(self, sig_fig = 2):
        print_list = [[round(x[0],sig_fig), round(x[1], sig_fig)] for x in self.linear_dims]
        return print_list
    
    def pick_new_points(self, num_samples, self_update = True, set_seed = None):
        '''Returns the index of points to get labels for, and automatically adds
        them the leafs labelled points by default
        '''

        num_samples = copy.copy(num_samples)

        if num_samples > len(self.unlabelled_index):
            raise ValueError('This leaf only has {} < {} unlabelled points'.format(
                len(self.unlabelled_index), num_samples))

        if set_seed is not None:
            random.seed(set_seed)

        new_points = random.sample(self.unlabelled_index, num_samples)
        if self_update:
            self.extend_labelled_index(new_points)
            self.unlabelled_index = [x for x in self.unlabelled_index if x not in new_points]

        return new_points

    def calculate_subtree_linear_dim(self):
        '''Since this 'subtree' is only the leaf, it's linear dim is just the
        sum of its linear dims.
        '''

        tot = 0
        for pair in self.linear_dims:
            tot += abs(pair[1] - pair[0])

        return tot

    def make_labelled(self, index):
        '''Takes a point in the unlablled_index list and moves it to the labelled one.
        If the point is not in the unlabelled_index list it returns an error.
        '''

        index = copy.copy(index)

        if index not in self.unlabelled_index:
            raise ValueError('Point {} is not in this leaf'.format(index))

        self.labelled_index.append(index)
        self.unlabelled_index.remove(index)

    ###########################################

    # Basic methods

    def is_leaf(self):
        return True

    def leaf_for_point(self, data_point):
        return self

    def extend_labelled_index(self, new_labelled_list):
        new_labelled_list = copy.deepcopy(new_labelled_list)
        self.labelled_index.extend(new_labelled_list)

    def extend_unlabelled_index(self, new_unlabelled_list):
        new_unlabelled_list = copy.deepcopy(new_unlabelled_list)
        self.unlabelled_index.extend(new_unlabelled_list)

    def set_linear_dims(self, new_linear_dims):
        new_linear_dims = copy.deepcopy(new_linear_dims)
        self.linear_dims = new_linear_dims
        self.subtree_linear_dim = self.calculate_subtree_linear_dim()

    def calculate_cell_l2_diameter(self):

        tot = 0
        for pair in self.linear_dims:
            tot += (pair[1] - pair[0])**2

        return math.sqrt(tot)
