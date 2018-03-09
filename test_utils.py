import unittest
import utils
import random

class test_utils(unittest.TestCase):

    # test choices

    def test_choices_empty(self):
        with self.assertRaises(ValueError):
            utils.choices([])

    def test_choices_uniform(self):
        random.seed(1)
        n = 10
        nreps = 1000
        counting_list = [0]*n
        for i in range(nreps):
            counting_list[utils.choices(range(n))[0]] += 1

        # print(counting_list)
        self.assertTrue(max(counting_list) - min(counting_list) < nreps/10)

    def test_choices_weighted(self):
        random.seed(1)
        n = 10
        nreps = 1000
        counting_list = [0]*n
        for i in range(nreps):
            counting_list[utils.choices(range(n), [1/(4*n)]*(n-1) + [1-(n-1)/(4*n)] )[0]] += 1

        # print(counting_list)
        self.assertTrue(counting_list[-1] > sum(counting_list) -  counting_list[-1])

    def test_choices_weighted_fully(self):
        random.seed(1)
        n = 10
        nreps = 1000
        counting_list = [0]*n
        for i in range(nreps):
            counting_list[utils.choices(range(n), [0]*(n-1) + [1] )[0]] += 1

        # print(counting_list)
        self.assertEqual(counting_list[-1], nreps)

    # test unbiased_var

    def test_unbaised_var_empty(self):
        self.assertEqual(utils.unbiased_var([]),0)

    def test_unbiased_var_single(self):
        self.assertEqual(utils.unbiased_var([1]),0)

    def test_unbiased_var_all_equal(self):
        self.assertEqual(utils.unbiased_var([1,1,1,1]),0)

    def test_unbaised_var_unequal(self):
        self.assertEqual(utils.unbiased_var([1,-1,1,-1]),4/3)

    def test_unbaised_var_unequal_2(self):
        self.assertEqual(utils.unbiased_var(range(10)),82.5/9)

if __name__ == '__main__':
    unittest.main()