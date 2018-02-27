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
        temp = Mondrian_Tree([[0,1]]*5)
        temp.update_life_time(0.1,set_seed=1)
        self.assertEqual(temp._num_leaves,3)

    def test_update_first_life_time_0p5(self):
        temp = Mondrian_Tree([[0,1]]*5)
        temp.update_life_time(0.5,set_seed=1)
        self.assertEqual(temp._num_leaves,8)

    def test_update_first_life_time_1(self):
        temp = Mondrian_Tree([[0,1]]*5)
        temp.update_life_time(1,set_seed=1)
        self.assertEqual(temp._num_leaves,36)

    def test_expected_split_bound(self):
        reps = 100
        tot = 0
        lbda = 0.5
        for i in range(reps):
            temp = Mondrian_Tree(self.linear_dims)
            temp.update_life_time(lbda,set_seed=i)
            tot += temp._num_leaves - 1
        # print(tot/reps, ((1+lbda)*math.exp(1))**self.d)
        self.assertTrue(tot/reps< ((1+lbda)*math.exp(1))**self.d)

    def test_make_full_leaf_list_base(self):
        self.mt1.make_full_leaf_list()
        self.assertEqual(len(self.mt1.full_leaf_list),1)

    def test_make_full_leaf_list(self):
        lbda = 1
        self.mt1.update_life_time(lbda,set_seed=1)
        self.mt1.make_full_leaf_list()
        # print(len(self.mt1.full_leaf_list))
        self.assertEqual(len(self.mt1.full_leaf_list),self.mt1._num_leaves)


    def test_expected_cell_diameter(self):
        tot = 0
        lbda = 3
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.make_full_leaf_list()
        for node in self.mt1.full_leaf_list:
            tot += node.calculate_cell_l2_diameter()**2

        # print(tot/len(self.mt1.full_leaf_list), 4*self.d/lbda**2)
        self.assertTrue(tot/len(self.mt1.full_leaf_list) < 4*self.d/lbda**2)





    # def test_update_life_time_tuned(self):

    #     self.mt1.update_life_time(self.a * self.n**(1/(2+self.d)),set_seed=4)
    #     self.assertEqual(self.mt1._num_leaves,1)



if __name__ == '__main__':
    unittest.main()