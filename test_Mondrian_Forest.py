import unittest
import math
import random
import warnings
import utils

import numpy as np

from Mondrian_Forest import Mondrian_Forest
from Mondrian_Tree import Mondrian_Tree
from LeafNode import LeafNode

class test_Mondrian_Forest(unittest.TestCase):

    def setUp(self):
        self.d = 3
        self.n_tree = 10
        self.linear_dims = [[0,1]]*self.d
        self.mf = Mondrian_Forest(self.linear_dims, self.n_tree)

        self.n_points = 100
        self.n_labelled = 20
        random.seed(123)
        self.labels = [random.random() for i in range(self.n_labelled)]
        self.labelled_indicies = range(self.n_labelled)
        self.data = []
        random.seed(1)
        for i in range(self.n_points):
            point = []
            for j in range(self.d):
                point.append(random.random())
            self.data.append(point)

    def test_building_forest(self):
        self.assertTrue(len(self.mf.tree_list),self.n_tree)
        self.assertFalse(self.mf.tree_list[0] == self.mf.tree_list[1])

    ###########################################

    # testing lifetime updates

    def test_update_life_time(self):
        lbda = 1
        self.mf.update_life_time(lbda, set_seeds = list(range(self.n_tree)))
        tree = Mondrian_Tree(self.linear_dims)
        tree.update_life_time(lbda, set_seed=0)
        self.assertEqual(tree._num_leaves, self.mf.tree_list[0]._num_leaves)
        for tree in self.mf.tree_list:
            self.assertEqual(tree._life_time, lbda)

    ###########################################

    # testing adding / updating data

    def test_input_data(self):
        self.mf.input_data(self.data, self.labelled_indicies, self.labels)
        for tree in self.mf.tree_list:
            self.assertTrue(self.mf.points is tree.points)
            self.assertEqual(self.mf._num_points, tree._num_points)

    def test_input_data_grow(self):
        lbda = 1
        self.mf.input_data(self.data, self.labelled_indicies, self.labels)
        self.mf.update_life_time(lbda, set_seeds = list(range(self.n_tree)))
        for i, tree in enumerate(self.mf.tree_list):
            test_tree = Mondrian_Tree(self.linear_dims)
            test_tree.update_life_time(lbda, set_seed=i)
            test_tree.input_data(self.data, self.labelled_indicies, self.labels)
            self.assertEqual(test_tree._num_leaves, tree._num_leaves)
            tree.make_full_leaf_list()
            test_tree.make_full_leaf_list()
            for j, leaf in enumerate(tree._full_leaf_list):
                self.assertEqual(
                    tree._full_leaf_list[j].labelled_index,
                    test_tree._full_leaf_list[j].labelled_index
                    )
                self.assertEqual(
                    tree._full_leaf_list[j].unlabelled_index,
                    test_tree._full_leaf_list[j].unlabelled_index
                    )

    def test_label_point(self):
        val = 2
        self.mf.input_data(self.data, self.labelled_indicies, self.labels)
        self.mf.label_point(self.n_labelled, val)
        self.assertEqual(self.mf.labels[self.n_labelled],val)
        self.assertEqual(self.mf._num_labelled, self.n_labelled + 1)
        for tree in self.mf.tree_list:
            leaf = tree._root.leaf_for_point(self.data[self.n_labelled])
            self.assertTrue(self.n_labelled in leaf.labelled_index)

    def test_add_data_point(self):
        lbda = 1
        self.mf.input_data(self.data, self.labelled_indicies, self.labels)
        self.mf.update_life_time(lbda, set_seeds = list(range(self.n_tree)))
        self.mf.add_data_point([1]*self.d)
        self.assertEqual(self.mf._num_points, self.n_points + 1)
        self.assertEqual(self.data + [[1]*self.d], self.mf.points)
        self.assertEqual(self.labels + [None] * (self.n_points - self.n_labelled) + 
            [None], self.mf.labels)
        for tree in self.mf.tree_list:
            leaf = tree._root.leaf_for_point([1]*self.d)
            self.assertEqual(leaf.unlabelled_index[-1],self.n_points)
            self.assertEqual(len(tree.points), tree._num_points)

    def test_add_data_point_labelled(self):
        lbda = 1
        val = 1
        self.mf.input_data(self.data, self.labelled_indicies, self.labels)
        self.mf.update_life_time(lbda, set_seeds = list(range(self.n_tree)))
        self.mf.add_data_point([1]*self.d, val)
        self.assertEqual(self.mf._num_points, self.n_points + 1)
        self.assertEqual(self.mf._num_labelled, self.n_labelled + 1)
        self.assertEqual(self.data + [[1]*self.d], self.mf.points)
        self.assertEqual(self.labels + [None] * (self.n_points - self.n_labelled) + 
            [val], self.mf.labels)
        for tree in self.mf.tree_list:
            leaf = tree._root.leaf_for_point([1]*self.d)
            self.assertEqual(leaf.labelled_index[-1],self.n_points)
            self.assertEqual(len(tree.points), tree._num_points)


if __name__ == '__main__':
    unittest.main()