import unittest
import copy
from LeafNode import LeafNode
from SplitNode import SplitNode

class test_SplitNode(unittest.TestCase):

    '''Unit testing for SplitNode.

    Tree 1 looks like this:

             A
            / \
           B   C
              / \
             D   E
            / \
           F   G

    And the data points are an equally spaced grid on [0,1] \times [0,1] like 
     1  2    3  4
     5  6    7  8

     9 10   11 12
    13 14   15 16

    where up-down is dim 0 and left-right is dim 1. So imagine point 1 is [0,1],
    point 4 is [1,1], point 7 is [0.75,0.75]. We put in no points on the [0.5,-], [-,0.5]
    lines to keep numbers round. Labelled and unlabelled are arbitrary.
    '''

    def setUp(self):

        self.G = LeafNode(
            labelled_index=[1],
            unlabelled_index=[2,5,6],
            linear_dims=[[0.7,1],[0,0.5]])
        self.F = LeafNode(
            labelled_index=[9],
            unlabelled_index=[11],
            linear_dims=[[0.2,0.7],[0,0.5]])
        self.D = SplitNode(split_dim = 0, split_val = 0.7,
            left_child=self.F,right_child=self.G)
        self.G.parent_node = self.D
        self.F.parent_node = self.D

        self.E = LeafNode(
            labelled_index=[11,4],
            unlabelled_index=[3,7,8,12],
            linear_dims=[[0.2,1],[0.5,1]])
        self.C = SplitNode(split_dim = 1, split_val = 0.5,
            left_child = self.D, right_child=self.E)
        self.E.parent_node = self.C
        self.D.parent_node = self.C

        self.B = LeafNode(
            labelled_index=[13],
            unlabelled_index=[14,15,16],
            linear_dims=[[0,0.2],[0,1]])
        self.A = SplitNode(split_dim = 0 ,split_val = 0.2,
            left_child=self.B, right_child=self.C)
        self.C.parent_node = self.A
        self.B.parent_node = self.A

    def test_calculate_entire_tree_linear_dim(self):
        self.assertEqual(self.A.calculate_subtree_linear_dim(),4.3)

    def test_leaf_for_point_in_domain(self):
        self.assertEqual(self.A.leaf_for_point([0,1]), self.B)

    def test_leaf_for_point_outof_domain(self):
        self.assertEqual(self.A.leaf_for_point([-1,2]), self.B)

    def test_percolate_subtree_linear_dim(self):
        self.A.calculate_subtree_linear_dim()
        self.D.percolate_subtree_linear_dim_change(0.5)
        self.assertEqual(self.A.subtree_linear_dim,4.8)

    def test_percolate_subtree_linear_dim__2(self):
        self.A.calculate_subtree_linear_dim()
        # This won't actually partition the space but is just a test of percolate
        temp_lin_dim = self.F.subtree_linear_dim
        self.F.set_linear_dims([[0,1],[0,1]])
        self.D.percolate_subtree_linear_dim_change(2 - temp_lin_dim)
        perc_lin_dim = copy.copy(self.A.subtree_linear_dim)
        # print(perc_lin_dim)
        self.A.calculate_subtree_linear_dim()
        # print(self.A.subtree_linear_dim)
        self.assertEqual(self.A.subtree_linear_dim, perc_lin_dim)

if __name__ == '__main__':
    unittest.main()