import random
import copy
import utils

from LeafNode import LeafNode
from SplitNode import SplitNode

class Mondrian_Tree:

    '''
    Main class for Mondrian Trees

    Args:
        linear_dims (list): A p dim list of 2 dim lists indicating the upper and 
        lower bounds of the entire space. New data points should be able to take points
        outside this space (probably!) but no partitions will take place outside this
        space so any partitioning will just be from infinite continuations of edge 
        partitions.
    '''

    def __init__(self, linear_dims):
        self._max_linear_dims = linear_dims
        self._root = LeafNode(linear_dims = self._max_linear_dims)
        self._num_dimensions = len(linear_dims)

        self.points = None
        self.labels = None
        self._num_points = 0

        self._life_time = 0
        self._num_leaves = 1

        self._full_leaf_list = []
        self._full_leaf_mean_list = []
        self._full_leaf_var_list = []
        self._full_leaf_marginal_list = []
        self._full_leaf_list_up_to_date = False

        self._verbose = False # useful for debugging or seeing how things work

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

    def update_life_time(self, new_life_time, set_seed = None):

        '''Function for updating the tree with a new life time parameter, potentially 
        growing the tree.
        '''

        if new_life_time < self._life_time:
            raise ValueError('The new life time {} must be larger than the old one {}.\
                This implementation does not support pruning of trees'.format(
                new_life_time, self._life_time))

        old_life_time = self._life_time
        self._life_time = new_life_time

        # Indicating we are growing our tree so the full leaf list will be wrong

        self._full_leaf_list = []
        self._full_leaf_list_up_to_date = False

        # We add new splits until the next split is after the new life time

        if set_seed is not None:
            random.seed(set_seed)

        next_split_time = old_life_time + random.expovariate(self._root.subtree_linear_dim)
        while next_split_time < self._life_time:

            # We need to pick which leaf to split. We move down the tree, moving left or 
            # right proportional to the linear_dim of all leaves in that subtree 
            # which is the subtree_linear_dim parameter of each node.

            self._num_leaves += 1

            curr_node = self._root
            while not curr_node.is_leaf():

                left_prob = curr_node.left_child.subtree_linear_dim
                right_prob = curr_node.right_child.subtree_linear_dim

                left_prob = left_prob / (left_prob + right_prob)
                right_prob = right_prob / (left_prob + right_prob)

                rand_split_val = random.random()

                if self._verbose:
                    print(
                        'Probability of going left is {}\n\
                        Probability of going right is {}\n\
                        random value is {}').format(left_prob, right_prob, rand_split_val)

                if rand_split_val < left_prob:
                    curr_node = curr_node.left_child
                    if self._verbose:
                        print('Going left')

                else:
                    curr_node = curr_node.right_child
                    if self._verbose:
                        print('Going right')

            # Now that we're at the leaf we are going to split, we need to split this leaf

            dimension_probs = []
            for pair in curr_node.linear_dims:
                dimension_probs.append(abs(pair[1] - pair[0])/curr_node.subtree_linear_dim)

            split_dim = utils.choices(range(self._num_dimensions), weights=dimension_probs)[0]
            split_interval = curr_node.linear_dims[split_dim]
            split_val = random.uniform(split_interval[0], split_interval[1])

            left_linear_dims = copy.deepcopy(curr_node.linear_dims)
            left_linear_dims[split_dim] = [split_interval[0],split_val]
            right_linear_dims = copy.deepcopy(curr_node.linear_dims)
            right_linear_dims[split_dim] = [split_val,split_interval[1]]

            # Build the new split and leaf nodes

            new_left_node = LeafNode(linear_dims = left_linear_dims, parent_branch = 0)
            new_right_node = LeafNode(linear_dims = right_linear_dims, parent_branch = 1)
            new_split_node = SplitNode(
                split_dim = split_dim,
                split_val = split_val,
                left_child = new_left_node,
                right_child = new_right_node,
                parent_node = curr_node.parent_node,
                parent_branch = curr_node.parent_branch,
                subtree_linear_dim = curr_node.subtree_linear_dim) # We will update subtree_lin_dim with percolate

            new_split_node.left_child.parent_node = new_split_node
            new_split_node.right_child.parent_node = new_split_node

            # Putting the new nodes into the tree

            if curr_node.parent_node is not None:
                if curr_node.parent_branch == 0:
                    curr_node.parent_node.left_child = new_split_node
                else:
                    curr_node.parent_node.right_child = new_split_node

            else:
                self._root = new_split_node

            # Percolating up the change in subtree_lin_dim

            subtree_lin_dim_change = (
                new_left_node.subtree_linear_dim + 
                new_right_node.subtree_linear_dim -
                curr_node.subtree_linear_dim)

            new_split_node.percolate_subtree_linear_dim_change(subtree_lin_dim_change)

            # moving data points into the new leaves

            for ind in curr_node.labelled_index:
                new_split_node.leaf_for_point(self.points[ind]).extend_labelled_index(ind)

            for ind in curr_node.unlabelled_index:
                new_split_node.leaf_for_point(self.points[ind]).extend_unlabelled_index(ind)

            next_split_time = next_split_time + random.expovariate(self._root.subtree_linear_dim)

    def input_data(all_data, labelled_indicies, labels):
        '''Puts in data for Mondrian Tree. 
        all_data should be a list of lists (or numpy array, points by row) with all data points, 
        labelled_indicies should be a list of the indicies for data points which we have the
        labels for, and labels should be an equal length list of those points labels.
        '''

        if len(all_data) < len(labelled_indicies):
            raise ValueError('Cannot have more labelled indicies than points')

        if len(labelled_indicies) != len(labels):
            raise ValueError('Labelled indicies list and labels list must be same length')

        self.points = all_data
        self._num_points = len(self.points)
        temp = [None] * self._num_points
        for i,ind in enumerate(labelled_indicies):
            temp[ind] = labels[i]
        self.labels = temp
        unlabelled_indicies = [x for x in range(self._num_points) if x not in labelled_indicies]

        if self._root.is_leaf():
            self._root.labelled_index = labelled_indicies
            self._root.unlabelled_index = unlabelled_indicies

        else:
            for i in labelled_indicies:
                curr_leaf = self._root.leaf_for_point(self.points[i])
                curr_leaf.labelled_index.append(i)

            for i in unlabelled_indicies:
                curr_leaf = self._root.leaf_for_point(self.points[i])
                curr_leaf.unlabelled_index.append(i)


    def make_full_leaf_list(self):
        '''Makes a list with pointers to every leaf in the tree. Likely to be expensive so 
        only do this if you're pre-building a tree for extensive use later. 
        '''

        full_leaf_list = []
        def internal_dfs(node):
            if node.is_leaf():
                full_leaf_list.append(node)
            else:
                internal_dfs(node.left_child)
                internal_dfs(node.right_child)

        internal_dfs(self._root)
        self._full_leaf_list = full_leaf_list
        for i, node in enumerate(self._full_leaf_list):
            node.full_leaf_list_pos = i
        self._full_leaf_list_up_to_date = True

    def make_full_leaf_var_list(self):
        if not self._full_leaf_list_up_to_date:
            print('Making full leaf list. Please wait')
            self.make_full_leaf_list()
            print('Done!')

        for i, node in enumerate(self._full_leaf_list):
            label_list = [self.labels[x] for x in node.labelled_index]
            self._full_leaf_var_list[i] = utils.unbiased_var(label_list)







