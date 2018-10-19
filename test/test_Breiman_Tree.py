import unittest
import random

from Breiman_Tree import Breiman_Tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class test_Breiman_Tree(unittest.TestCase):

    def setUp(self):

        self.d = 10
        self.a = 1
        self.bt = Breiman_Tree(min_samples_leaf=5)

        self.n_points = 100
        self.n_labelled = 20
        random.seed(123)
        self.labels = [random.random() for i in range(self.n_labelled)]
        self.labelled_indices = range(self.n_labelled)
        self.data = []
        random.seed(1)
        for i in range(self.n_points):
            point = []
            for j in range(self.d):
                point.append(random.random())
            self.data.append(point)

    def test_building_tree_with_same_data(self):
        n_reps = 20
        self.bt.input_data(self.data, self.labelled_indices, self.labels)
        tree_lens = []
        for _ in range(n_reps):
            self.bt.fit_tree()
            tree_lens.append(self.bt.tree.tree_.node_count)
            # print(self.bt.tree.tree_.node_count)
        self.assertEqual([tree_lens[0]]*n_reps, tree_lens)

    def test_building_tree_with_incremental_data(self):
        '''No actual tests just printing things to look at how Breiman trees behave'''
        n_reps = 20
        self.bt.input_data(self.data, self.labelled_indices, self.labels)
        tree_lens = []
        for i in range(n_reps):
            self.bt.label_point(self.n_labelled+i, random.random())
            self.bt.fit_tree()
            # print(np.unique(self.bt.tree.tree_.threshold))
            tree_lens.append(len([x for x in self.bt.tree.tree_.children_left if x == -1]))
            
            # print(self.bt.tree.tree_.node_count)
        # print(tree_lens)

    def test_leaf_statistics_var(self):
        n_reps = 20
        self.bt.input_data(self.data, self.labelled_indices, self.labels)
        for i in range(n_reps):
            self.bt.label_point(self.n_labelled+i, random.random())
            self.bt.fit_tree()
            self.bt.calculate_leaf_statistics()
            # print(sorted(self.bt._leaf_var))

    def test_leaf_statistics_marginal(self):
        n_reps = 20
        self.bt.input_data(self.data, self.labelled_indices, self.labels)
        for i in range(n_reps):
            self.bt.label_point(self.n_labelled+i, random.random())
            self.bt.fit_tree()
            self.bt.calculate_leaf_statistics()
            # print(sorted(self.bt._leaf_marginal))

    def test_leaf_proportions(self):
        n_reps = 20
        self.bt.input_data(self.data, self.labelled_indices, self.labels)
        for i in range(n_reps):
            self.bt.label_point(self.n_labelled+i, random.random())
            self.bt.fit_tree()
            self.bt.calculate_leaf_statistics()
            self.bt.al_calculate_leaf_proportions()

    def test_pick_new_points(self):
        n_reps = 20
        self.bt.input_data(self.data, self.labelled_indices, self.labels)
        new_point = self.n_labelled
        for i in range(n_reps):
            self.bt.label_point(new_point, random.random())
            self.bt.fit_tree()
            self.bt.calculate_leaf_statistics()
            self.bt.al_calculate_leaf_proportions()
            new_point = self.bt.pick_new_points()[0]
            print(new_point)

        