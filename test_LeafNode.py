import unittest
from LeafNode import LeafNode

class test_LeafNode(unittest.TestCase):

    def setUp(self):
        self.test_leaf_empty = LeafNode()
        self.test_leaf_good = LeafNode([1,2,3,4], [5,6,7,8,9,10], [[0,1], [0,1]])

    # Testing the subtree_linear_dim method

    def test_dim_length_good(self):
        self.assertEqual(self.test_leaf_good.get_subtree_linear_dim(), 2)

    def test_dim_length_empty(self):
        self.assertEqual(self.test_leaf_empty.get_subtree_linear_dim(), 0)

    # Testing the pick_new_points method

    def test_pick_new_points_good(self):
        self.assertEqual(self.test_leaf_good.pick_new_points(1, set_seed=1),[6])
        self.assertEqual(self.test_leaf_good.pick_new_points(2, set_seed=1),[7,5])

    def test_pick_new_points_good_all(self):
        self.assertEqual(self.test_leaf_good.pick_new_points(6, set_seed=1),[6,9,5,10,8,7])

    def test_pick_new_points_empty(self):
        with self.assertRaises(ValueError):
            self.test_leaf_empty.pick_new_points(1)

    def test_pick_new_points_self_update_true(self):
        self.test_leaf_good.pick_new_points(1, set_seed=1)
        self.assertEqual(self.test_leaf_good.get_labelled_index(),[1,2,3,4,6])
        self.assertEqual(self.test_leaf_good.get_unlabelled_index(),[5,7,8,9,10])

    def test_pick_new_points_self_update_false(self):
        self.test_leaf_good.pick_new_points(1, self_update = False, set_seed=1)
        self.assertEqual(self.test_leaf_good.get_labelled_index(),[1,2,3,4])
        self.assertEqual(self.test_leaf_good.get_unlabelled_index(),[5,6,7,8,9,10])

if __name__ == '__main__':
    unittest.main()