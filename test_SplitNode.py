import unittest
from LeafNode import LeafNode
from SplitNode import SplitNode

class test_SplitNode(unittest.TestCase):

    def setUp(self):
        self.test_leaf_empty = LeafNode()
        self.test_leaf_good = LeafNode([1,2,3,4], [5,6,7,8,9,10], [[0,1], [0,1]])

if __name__ == '__main__':
    unittest.main()