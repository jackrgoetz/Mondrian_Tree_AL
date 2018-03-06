import unittest
import math
import random
import warnings
import utils

from Mondrian_Tree import Mondrian_Tree
from LeafNode import LeafNode


class test_Mondrian_Tree(unittest.TestCase):

    # TODO: Run tests with inputing data with numpy arrays

    def setUp(self):

        self.d = 3
        self.a = 1
        self.linear_dims = [[0,1]]*self.d
        self.mt1 = Mondrian_Tree(self.linear_dims)

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

    ###########################################

    # Basic life time updating tests

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

    def test_update_life_time_tuned(self):

        self.mt1.update_life_time(self.a * self.n_labelled**(1/(2+self.d) - 1),set_seed=1)
        self.assertEqual(self.mt1._num_leaves,2)

    # Testing input data

    def test_input_data_empty_root(self):
        self.mt1.input_data([],[],[])
        self.assertEqual(self.mt1._num_points,0)

    def test_input_data_empty(self):
        lbda = 1
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.input_data([],[],[])
        self.assertEqual(self.mt1._num_points,0)

    def test_input_data_root_no_labels(self):
        self.mt1.input_data([[0]*self.d],[],[])
        self.assertEqual(self.mt1._num_points,1)
        self.assertEqual(self.mt1.labels,[None])
        self.assertEqual(self.mt1._root.unlabelled_index,[0])
        self.assertEqual(self.mt1._root.labelled_index,[])

    def test_input_data_root_labels(self):
        self.mt1.input_data([[0]*self.d],[0],[3.141])
        self.assertEqual(self.mt1._num_points,1)
        self.assertEqual(self.mt1.labels,[3.141])
        self.assertEqual(self.mt1._root.unlabelled_index,[])
        self.assertEqual(self.mt1._root.labelled_index,[0])

    def test_input_data(self):
        lbda = 0.5
        self.mt1.update_life_time(lbda, set_seed=100)
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)

        self.assertEqual(self.mt1._num_points,self.n_points)
        self.assertEqual(self.mt1._num_labelled,self.n_labelled)
        
        self.mt1.make_full_leaf_list()
        for node in self.mt1._full_leaf_list:
            # print(len(node.labelled_index), len(node.unlabelled_index))

            linear_dims = node.linear_dims
            for ind in node.labelled_index:
                point = self.mt1.points[ind]
                for dim in range(self.d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

            for ind in node.unlabelled_index:
                point = self.mt1.points[ind]
                for dim in range(self.d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

    def test_update_life_time_with_data(self):
        lbda = 0.5
        lbda2 = 1
        self.mt1.update_life_time(lbda, set_seed=100)
        # print(self.mt1._num_points)

        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.update_life_time(lbda2, set_seed=100)
        # print(self.mt1._num_points)

        self.mt1.make_full_leaf_list()
        for node in self.mt1._full_leaf_list:
            # print(len(node.labelled_index), len(node.unlabelled_index))

            linear_dims = node.linear_dims
            for ind in node.labelled_index:
                point = self.mt1.points[ind]
                for dim in range(self.d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

            for ind in node.unlabelled_index:
                point = self.mt1.points[ind]
                for dim in range(self.d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

    def test_label_point_root(self):
        val = 1
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.label_point(self.n_labelled, val)
        self.assertEqual(self.mt1.labels[self.n_labelled],val)
        self.assertEqual(self.mt1._num_labelled, self.n_labelled + 1)
        leaf = self.mt1._root.leaf_for_point(self.data[self.n_labelled])
        self.assertTrue(self.n_labelled in leaf.labelled_index)

    def test_label_point_empty(self):
        self.assertTrue(False, 'Next test to write')

    def test_label_point(self):
        val = 1
        lbda = 0.5
        seed = 1
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.update_life_time(lbda, set_seed=seed)
        self.mt1.label_point(self.n_labelled, val)
        self.assertEqual(self.mt1.labels[self.n_labelled],val)
        self.assertEqual(self.mt1._num_labelled, self.n_labelled + 1)
        leaf = self.mt1._root.leaf_for_point(self.data[self.n_labelled])
        # print(leaf.labelled_index)
        self.assertTrue(self.n_labelled in leaf.labelled_index)

    ###########################################

    # Testing make leaf list 

    def test_make_full_leaf_list_root(self):
        self.mt1.make_full_leaf_list()
        self.assertEqual(len(self.mt1._full_leaf_list),1)

    def test_make_full_leaf_list(self):
        lbda = 1
        self.mt1.update_life_time(lbda,set_seed=1)
        self.mt1.make_full_leaf_list()
        # print(len(self.mt1._full_leaf_list))
        # for node in self.mt1._full_leaf_list:
        #     print(node.leaf_id)
        self.assertEqual(len(self.mt1._full_leaf_list),self.mt1._num_leaves)

    # Testing calculating leaf variances

    def test_make_full_leaf_var_list_root(self):
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_var_list()
        self.assertEqual(utils.unbiased_var(self.labels), self.mt1._full_leaf_var_list[0])

    def test_make_full_leaf_var_list_empty(self):
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_var_list()
        self.assertEqual(0, self.mt1._full_leaf_var_list[0])

    def test_make_full_leaf_var_list(self):
        lbda = 0.5
        self.mt1.update_life_time(lbda, set_seed=100)

        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_var_list()

        for i, node in enumerate(self.mt1._full_leaf_list):
            node_labels = [self.mt1.labels[x] for x in node.labelled_index]
            # print(node_labels)
            if len(node_labels) != 0:
                temp_mean = sum(node_labels)/len(node_labels)
                # print(temp_mean)
                temp_var = 1/(len(node_labels)-1) * sum([(x-temp_mean)**2 for x in node_labels])
                self.assertTrue(abs(self.mt1._full_leaf_var_list[i] - temp_var) < 1e-9)
            else:
                self.assertEqual(self.mt1._full_leaf_var_list[i],0)

    # Testing calculating leaf mean

    def test_make_full_leaf_mean_list_root(self):
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_mean_list()
        self.assertEqual(sum(self.labels)/self.n_labelled, self.mt1._full_leaf_mean_list[0])

    def test_make_full_leaf_mean_list_empty(self):
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_mean_list()
        self.assertEqual(0, self.mt1._full_leaf_mean_list[0])

    def test_make_full_leaf_mean_list(self):
        lbda = 0.5
        self.mt1.update_life_time(lbda, set_seed=100)

        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_mean_list()

        for i, node in enumerate(self.mt1._full_leaf_list):
            node_labels = [self.mt1.labels[x] for x in node.labelled_index]
            # print(node_labels)
            if len(node_labels) != 0:
                temp_mean = sum(node_labels)/len(node_labels)
                # print(temp_mean)
                self.assertTrue(abs(self.mt1._full_leaf_mean_list[i] - temp_mean) < 1e-9)
            else:
                self.assertEqual(self.mt1._full_leaf_mean_list[i],0)

    # Testing calculating leaf marginal probabilities

    def test_make_full_leaf_marginal_list_root(self):
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_marginal_list()
        self.assertEqual(1, self.mt1._full_leaf_marginal_list[0])

    def test_make_full_leaf_marginal_list_empty(self):
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_marginal_list()
        self.assertEqual(0, self.mt1._full_leaf_marginal_list[0])

    def test_make_full_leaf_marginal_list(self):
        lbda = 0.5
        self.mt1.update_life_time(lbda, set_seed=100)

        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.make_full_leaf_list()
        self.mt1.make_full_leaf_marginal_list()

        for i, node in enumerate(self.mt1._full_leaf_list):
            node_points = (
                [self.mt1.points[x] for x in node.labelled_index]+
                [self.mt1.points[x] for x in node.unlabelled_index])
            # print(node_labels)
            temp_marginal = len(node_points)/self.mt1._num_points
            self.assertTrue(abs(self.mt1._full_leaf_marginal_list[i] - temp_marginal) < 1e-9)

    ###########################################

    # Testing using predict and get_point_in_same_leaf

    def test_predict_and_get_point_in_same_leaf_bad_input(self):
        with self.assertRaises(TypeError):
            self.mt1.predict(1)
            self.mt1.get_points_in_same_leaf(1)

    def test_predict_and_get_point_in_same_leaf_bad_length(self):
        with self.assertRaises(ValueError):
            self.mt1.predict([])
            self.mt1.get_points_in_same_leaf([])

    def test_predict_empty(self):
        with self.assertWarns(UserWarning):
            self.mt1.predict([0.5]*self.d)

    def test_predict_no_leaf_list(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=100)
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)

        random.seed(seed)
        new_point = []
        for j in range(self.d):
            new_point.append(random.random())
        pred = self.mt1.predict(new_point)
        # print(pred)

        node = self.mt1._root.leaf_for_point(new_point)
        vals = [self.labels[x] for x in node.labelled_index]
        # print(len(vals))
        self.assertEqual(pred, sum(vals)/len(vals))

    def test_predict_leaf_list(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=100)
        self.mt1.make_full_leaf_list()
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)

        random.seed(seed)
        new_point = []
        for j in range(self.d):
            new_point.append(random.random())
        pred = self.mt1.predict(new_point)
        # print(pred)

        node = self.mt1._root.leaf_for_point(new_point)
        vals = [self.labels[x] for x in node.labelled_index]
        # print(len(vals))
        self.assertEqual(pred, sum(vals)/len(vals))

    def test_get_point_in_same_leaf_labelled(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=100)
        self.mt1.make_full_leaf_list()
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)

        random.seed(seed)
        new_point = []
        for j in range(self.d):
            new_point.append(random.random())
        labelled_list = self.mt1.get_points_in_same_leaf(new_point)

        node = self.mt1._root.leaf_for_point(new_point)
        self.assertEqual(labelled_list,node.labelled_index)

    def test_get_point_in_same_leaf_unlabelled(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=100)
        self.mt1.make_full_leaf_list()
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)

        random.seed(seed)
        new_point = []
        for j in range(self.d):
            new_point.append(random.random())
        unlabelled_list = self.mt1.get_points_in_same_leaf(new_point, 'unlabelled')

        node = self.mt1._root.leaf_for_point(new_point)
        self.assertEqual(unlabelled_list,node.unlabelled_index)

    def test_get_point_in_same_leaf_bad_list_name(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=100)
        self.mt1.make_full_leaf_list()
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)

        random.seed(seed)
        new_point = []
        for j in range(self.d):
            new_point.append(random.random())
        with self.assertWarns(UserWarning):
            labelled_list = self.mt1.get_points_in_same_leaf(new_point, 'neither')

        node = self.mt1._root.leaf_for_point(new_point)
        self.assertEqual(labelled_list,node.labelled_index)

    ###########################################


    # Testing theoretical bounds

    def test_expected_split_bound(self):
        reps = 100
        tot = 0
        lbda = 5
        d = 3
        for i in range(reps):
            temp = Mondrian_Tree([[0,1]]*d)
            temp.update_life_time(lbda,set_seed=i)
            tot += temp._num_leaves - 1
        # print(tot/reps, ((1+lbda)*math.exp(1))**d)
        self.assertTrue(tot/reps< ((1+lbda)*math.exp(1))**d)


    def test_expected_cell_diameter_bounds(self):
        tot = 0
        lbda = 5
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.make_full_leaf_list()
        for node in self.mt1._full_leaf_list:
            tot += node.calculate_cell_l2_diameter()**2

        # print(tot/len(self.mt1._full_leaf_list), 4*self.d/lbda**2)
        self.assertTrue(tot/len(self.mt1._full_leaf_list) < 4*self.d/lbda**2)

    def test_probabilistic_cell_diameter_bounds(self):
        lbda = 5
        deltas = [0.1,0.2,0.3,0.4,0.5,1,1.5]
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.make_full_leaf_list()
        # print(len(self.mt1._full_leaf_list))
        for delta in deltas:
            tot = 0
            for node in self.mt1._full_leaf_list:
                tot += int(node.calculate_cell_l2_diameter() > delta)

            # print(tot/len(self.mt1._full_leaf_list), 
            #     self.d * (1+ (lbda * delta)/math.sqrt(self.d))*
            #     math.exp(-lbda * delta/math.sqrt(self.d)))
            self.assertTrue(tot/len(self.mt1._full_leaf_list) < 
                self.d * (1+ (lbda * delta)/math.sqrt(self.d))*
                math.exp(-lbda * delta/math.sqrt(self.d)))

    ###########################################

    # Testing data driven default values for prediction and active learning variance

    def test_set_default_pred_global_mean_empty(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=seed)
        self.mt1.make_full_leaf_list()
        self.mt1.prediction_default_value = 1
        self.mt1.set_default_pred_global_mean()
        self.assertEqual(self.mt1.prediction_default_value, 0)

    def test_set_default_pred_global_mean(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=seed)
        self.mt1.make_full_leaf_list()
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.set_default_pred_global_mean()
        self.assertEqual(self.mt1.prediction_default_value, sum(self.labels)/self.n_labelled)

    def test_al_set_default_var_global_var_empty(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=seed)
        self.mt1.make_full_leaf_list()
        self.mt1.al_default_var = 1
        self.mt1.al_set_default_var_global_var()
        self.assertEqual(self.mt1.al_default_var, 0)

    def test_al_set_default_var_global_var(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=seed)
        self.mt1.make_full_leaf_list()
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.al_set_default_var_global_var()
        self.assertEqual(self.mt1.al_default_var, utils.unbiased_var(self.labels))

    ###########################################


    # Testing active learning parts

    def test_al_calculate_leaf_proportions_empty_root(self):
        with self.assertWarns(UserWarning):
            self.mt1.al_calculate_leaf_proportions()
        self.assertEqual(self.mt1._al_proportions, [1])

    def test_al_calculate_leaf_proportions_empty(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=seed)
        with self.assertWarns(UserWarning):
            self.mt1.al_calculate_leaf_proportions()
        self.assertEqual(self.mt1._al_proportions, [1/self.mt1._num_leaves]*self.mt1._num_leaves)

    def test_al_calculate_leaf_proportions_root(self):
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.al_calculate_leaf_proportions()
        self.assertEqual(self.mt1._al_proportions, [1])

    def test_al_calculate_leaf_proportions(self):
        lbda = 0.5
        seed = 1
        self.mt1.update_life_time(lbda, set_seed=seed)
        self.mt1.input_data(self.data, self.labelled_indicies, self.labels)
        self.mt1.al_calculate_leaf_proportions()
        temp_prop_list = []
        for i, node in enumerate(self.mt1._full_leaf_list):
            node_points = [x for x in node.labelled_index] + [x for x in node.unlabelled_index]
            # print(len(node_points))
            # print(self.mt1._full_leaf_marginal_list[i])
            if len(node_points) != 0:
                temp_var = utils.unbiased_var([self.labels[x] for x in node.labelled_index])
                # print(self.mt1._al_proportions[i])
                # print(math.sqrt(temp_var * (len(node_labels)/self.n_points)))
                temp_prop_list.append(math.sqrt(temp_var * (len(node_points)/self.n_points)))
                
            else:
                self.assertEqual(self.mt1._al_proportions[i],0)

        normalizer = sum(temp_prop_list)
        temp_prop_list = [x/normalizer for x in temp_prop_list]
        # print(temp_prop_list)
        for i, val in enumerate(temp_prop_list):
            # print(val, self.mt1._al_proportions[i])
            self.assertTrue(abs(self.mt1._al_proportions[i] - val) < 1e-9)






if __name__ == '__main__':
    unittest.main()