import random

class LeafNode:

    '''Nodes for tree leaves

    Args:
        labelled_index (list): Index for labelled data points in the leaf.
        unlabelled_index (list): Index for unlabelled data points in the leaf.
        linear_dims (list): A p dim list of tuples indicating the upper and 
        lower bounds of the leaf.
        parent_node (SplitNode): This node's parent
        parent_branch (int): This node's branch (0 for left, 1 for right)
    '''
    def __init__(self, labelled_index=[], unlabelled_index=[],linear_dims=[],
        parent_node=None, parent_branch=None):
        self.labelled_index = labelled_index
        self.unlabelled_index = unlabelled_index
        self.linear_dims = linear_dims

    def __str__(self):
        print_str = 'n_labelled = {}, n_unlabelled = {}, '.format(
            len(self.labelled_index), len(self.unlabelled_index))
        return(print_str)
    
    def pick_new_points(self, num_samples, self_update = True, set_seed = None):
        # Returns the index of points to get labels for, and automatically adds
        # them the leafs labelled points by default
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

    def get_subtree_linear_dim(self):
        # Since this 'subtree' is only the leaf, it's linear dim is just the
        # sum of its linear dims

        tot = 0
        for pair in self.linear_dims:
            tot += abs(pair[1] - pair[0])

        return tot

    ###########################################

    # Basic functions

    def is_leaf(self):
        return True

    def set_labelled_index(self, new_labelled_index):
        self.labelled_index = new_labelled_index

    def extend_labelled_index(self, new_labelled):
        self.labelled_index.extend(new_labelled)

    def get_labelled_index(self):
        return self.labelled_index

    def set_unlablled_index(self, new_unlablled_index):
        self.unlabelled_index = new_unlablled_index

    def extend_unlabllerd_index(self, new_unlabelled):
        self.unlabelled_index.extend(new_unlabelled)

    def get_unlabelled_index(self):
        return self.unlabelled_index

    def set_linear_dims(self, new_linear_dims):
        self.linear_dims = new_linear_dims

    def get_linear_dims(self):
        return self.linear_dims





