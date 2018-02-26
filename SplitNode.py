

class SplitNode:

    '''Nodes for the tree interior

    Args:
        split_dim (int): The dimension this node split on (starts at 0)
        split_val (float): The value of the split point
        parent_node (SplitNode): This node's parent node
        parent_branch (int): This node's branch (0 for left, 1 for right)
    '''

    def __init__(self, split_dim, split_val,left_child, right_child, 
        parent_node=None, parent_branch=None, subtree_linear_dim=None):
        self.split_dim = split_dim
        self.split_val = split_val
        self.parent_node = parent_node
        self.parent_branch = parent_branch
        self.left_child = left_child
        self.right_child = right_child
        self.subtree_linear_dim = subtree_linear_dim

    def __str__(self):
        return 'Splits on dimension {} and value {}'.format(self.split_dim, self.split_val)

    def calculate_subtree_linear_dim(self):
        '''Note this is likely to be very expensive. Ideally the subtree linear dim value
        should be updated as you build the tree percolating up
        '''

        self.subtree_linear_dim = (self.left_child.calculate_subtree_linear_dim() +
                                   self.right_child.calculate_subtree_linear_dim())
        return self.subtree_linear_dim

    def leaf_for_point(self, data_point):
        if data_point[self.split_dim] < self.split_val:
            if self.left_child.is_leaf():
                return self.left_child
            else:
                return self.left_child.leaf_for_point(data_point)

        else:
            if self.right_child.is_leaf():
                return self.right_child
            else:
                return self.right_child.leaf_for_point(data_point)

    ###########################################

    # Basic methods

    def is_leaf(self):
        return False