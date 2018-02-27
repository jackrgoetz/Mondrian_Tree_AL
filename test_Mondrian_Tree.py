import unittest
import math
from Mondrian_Tree import Mondrian_Tree

class test_Mondrian_Tree(unittest.TestCase):

    def setUp(self):

        self.d = 5
        self.n = 1000
        self.a = 1
        self.linear_dims = [[0,1]]*self.d
        self.mt1 = Mondrian_Tree(self.linear_dims)

    def test_update_first_life_time_0p1(self):
        self.assertEqual(self.mt1._num_leaves,1)
        self.mt1.update_life_time(0.1,set_seed=1)
        self.assertEqual(self.mt1._num_leaves,3)

    def test_update_first_life_time_0p5(self):
        self.assertEqual(self.mt1._num_leaves,1)
        self.mt1.update_life_time(0.5,set_seed=1)
        self.assertEqual(self.mt1._num_leaves,8)

    def test_update_first_life_time_1(self):
        self.assertEqual(self.mt1._num_leaves,1)
        self.mt1.update_life_time(1,set_seed=1)
        self.assertEqual(self.mt1._num_leaves,36)

    def test_expected_split_bound(self):
        reps = 100
        tot = 0
        lbda = 2
        for i in range(reps):
            temp = Mondrian_Tree(self.linear_dims)
            temp.update_life_time(1,set_seed=i)
            tot += temp._num_leaves - 1

        self.assertTrue(tot/reps< ((1+lbda)*math.exp(1))**self.d)





    # def test_update_life_time_tuned(self):

    #     self.mt1.update_life_time(self.a * self.n**(1/(2+self.d)),set_seed=4)
    #     self.assertEqual(self.mt1._num_leaves,1)



if __name__ == '__main__':
    unittest.main()