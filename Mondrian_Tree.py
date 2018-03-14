import random
import copy
import utils
import warnings
import math

from LeafNode import LeafNode
from SplitNode import SplitNode

class Mondrian_Tree:

    '''
    Main class for Mondrian Trees

    Args:
        linear_dims (list): A p dim list of 2 dim lists indicating the upper and 
        lower bounds of the entire space. New data points should be able to take points
        outside this space (probably!) but no partitions will take place outside this
        space so any partitioning will just be from infinite continuations of edge 
        partitions.
    '''

    def __init__(self, linear_dims):
        self._max_linear_dims = linear_dims
        self._root = LeafNode(linear_dims = self._max_linear_dims)
        self._num_dimensions = len(linear_dims)

        self.points = None
        self.labels = None
        self._num_points = 0
        self._num_labelled = 0

        self._life_time = 0
        self._num_leaves = 1

        self._full_leaf_list = []
        self._full_leaf_mean_list = []
        self._full_leaf_var_list = []
        self._full_leaf_marginal_list = []
        self._full_leaf_list_up_to_date = False
        self._full_leaf_mean_list_up_to_date = False
        self._full_leaf_var_list_up_to_date = False
        self._full_leaf_marginal_list_up_to_date = False

        self.prediction_default_value = 0

        self._al_proportions = []
        self._al_proportions_up_to_date = False
        self.al_default_var = 0
        self._al_point_weights_proportional = None
        self._al_point_weights_adjustment = None
        self._al_leaf_number_new_labels = None

        self._verbose = False # useful for debugging or seeing how things work

    def __str__(self):
        # Add more as needed
        return (
        'Number of dimensions = {}\n' 
        'Number of leaf nodes = {}\n'
        'Life time parameter = {}\n'
        '\n'
        'Number of data points = {}\n'
        'Number of labels = {}'.format(
            self._num_dimensions, self._num_leaves, self._life_time, self._num_points, 
            self._num_labelled))

    def _test_point(self, new_point):
        '''Tests an input point, raising errors if it's a bad type and converting it from
        a numpy array to a list if needed 
        '''

        try:
            len(new_point)
        except TypeError:
            raise TypeError(
                'Given point has no len(), so probably is not a vector representing a data point. '
                'Try turning it into a list, tuple or numpy array where each entry is a dimension.')

        if len(new_point) != self._num_dimensions:
            raise ValueError(
                'Data point is not of the correct length. Must be the same dimension as the '
                'dimensions used to build the Mondrian Tree when it was initialized.')

        if str(type(new_point)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting new_point to list internally')
            new_point = new_point.tolist()

        if type(new_point) not in [list, tuple]:
            raise TypeError('Please input the new point as a list, tuple or numpy array')

        return new_point

    ###########################################

    # Tree building and updating methods: Builds the tree based on the life time, and
    # put data into the tree. The tree is build completely independently of any data,
    # and can be built without having any data in it. The data can be added later 
    # and it will be put into the correct leaves and everything. 

    def update_life_time(self, new_life_time, set_seed = None):

        '''Function for updating the tree with a new life time parameter, potentially 
        growing the tree. Grows until the next split would occur after the new life
        time, moving any data within the tree into the new leaves.
        '''

        new_life_time = copy.copy(new_life_time)

        if new_life_time < self._life_time:
            raise ValueError('The new life time {} must be larger than the old one {}.\
                This implementation does not support pruning of trees'.format(
                new_life_time, self._life_time))

        old_life_time = self._life_time
        self._life_time = new_life_time

        # Indicating we are growing our tree so the full leaf list will be wrong

        self._full_leaf_list = []
        self._full_leaf_list_up_to_date = False
        self._full_leaf_mean_list_up_to_date = False
        self._full_leaf_var_list_up_to_date = False
        self._full_leaf_marginal_list_up_to_date = False
        self._active_learning_proportions_up_to_date = False

        # We add new splits until the next split is after the new life time

        if set_seed is not None:
            random.seed(set_seed)

        next_split_time = old_life_time + random.expovariate(self._root.subtree_linear_dim)
        while next_split_time < self._life_time:

            # We need to pick which leaf to split. We move down the tree, moving left or 
            # right proportional to the linear_dim of all leaves in that subtree 
            # which is the subtree_linear_dim parameter of each node.

            self._num_leaves += 1

            curr_node = self._root
            while not curr_node.is_leaf():

                left_prob = curr_node.left_child.subtree_linear_dim
                right_prob = curr_node.right_child.subtree_linear_dim

                left_prob = left_prob / (left_prob + right_prob)
                right_prob = right_prob / (left_prob + right_prob)

                rand_split_val = random.random()

                if self._verbose:
                    print(
                        'Probability of going left is {}\n\
                        Probability of going right is {}\n\
                        Random value is {}').format(left_prob, right_prob, rand_split_val)

                if rand_split_val < left_prob:
                    curr_node = curr_node.left_child
                    if self._verbose:
                        print('Going left')

                else:
                    curr_node = curr_node.right_child
                    if self._verbose:
                        print('Going right')

            # Now that we're at the leaf we are going to split, we need to split this leaf.
            # We pick the dimension to split on proportional to it's length, and then pick
            # a split point uniformly on that dimension

            dimension_probs = []
            for pair in curr_node.linear_dims:
                dimension_probs.append(abs(pair[1] - pair[0])/curr_node.subtree_linear_dim)

            split_dim = utils.choices(range(self._num_dimensions), weights=dimension_probs)[0]
            split_interval = curr_node.linear_dims[split_dim]
            split_val = random.uniform(split_interval[0], split_interval[1])

            left_linear_dims = copy.deepcopy(curr_node.linear_dims)
            left_linear_dims[split_dim] = [split_interval[0],split_val]
            right_linear_dims = copy.deepcopy(curr_node.linear_dims)
            right_linear_dims[split_dim] = [split_val,split_interval[1]]

            # Build the new split and leaf nodes

            new_left_node = LeafNode(linear_dims = left_linear_dims, parent_branch = 0)
            new_right_node = LeafNode(linear_dims = right_linear_dims, parent_branch = 1)
            new_split_node = SplitNode(
                split_dim = split_dim,
                split_val = split_val,
                left_child = new_left_node,
                right_child = new_right_node,
                parent_node = curr_node.parent_node,
                parent_branch = curr_node.parent_branch,
                subtree_linear_dim = curr_node.subtree_linear_dim) # We will update subtree_lin_dim with percolate

            new_split_node.left_child.parent_node = new_split_node
            new_split_node.right_child.parent_node = new_split_node

            # Putting the new nodes into the tree

            if curr_node.parent_node is not None:
                if curr_node.parent_branch == 0:
                    curr_node.parent_node.left_child = new_split_node
                else:
                    curr_node.parent_node.right_child = new_split_node

            else:
                self._root = new_split_node

            # Percolating up the change in subtree_lin_dim

            subtree_lin_dim_change = (
                new_left_node.subtree_linear_dim + 
                new_right_node.subtree_linear_dim -
                curr_node.subtree_linear_dim)

            new_split_node.percolate_subtree_linear_dim_change(subtree_lin_dim_change)

            # moving data points into the new leaves

            for ind in curr_node.labelled_index:
                # print(curr_node.labelled_index)
                new_split_node.leaf_for_point(self.points[ind]).extend_labelled_index([ind])

            for ind in curr_node.unlabelled_index:
                new_split_node.leaf_for_point(self.points[ind]).extend_unlabelled_index([ind])

            next_split_time = next_split_time + random.expovariate(self._root.subtree_linear_dim)

    def input_data(self, all_data, labelled_indicies, labels):
        '''Puts in data for Mondrian Tree. 
        all_data should be a list of lists (or numpy array, points by row) with all data points, 
        labelled_indicies should be a list of the indicies for data points which we have the
        labels for, and labels should be an equal length list of those points labels.

        Should work with inputting things as numpy arrays, but this is the only place you can 
        safely use numpy arrays. 
        '''

        all_data = copy.deepcopy(all_data)
        labelled_indicies = copy.deepcopy(labelled_indicies)
        labels = copy.deepcopy(labels)

        if len(all_data) < len(labelled_indicies):
            raise ValueError('Cannot have more labelled indicies than points')

        if len(labelled_indicies) != len(labels):
            raise ValueError('Labelled indicies list and labels list must be same length')

        for point in all_data:
            if len(point) != self._num_dimensions:
                raise ValueError('All data points must be of the dimension on which this \
                    Mondrian Tree is built ({})'.format(self._num_dimensions))

        if str(type(all_data)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting all_data to list of lists internally')
            all_data = all_data.tolist()

        if str(type(labelled_indicies)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labelled_indicies to list internally')
            labelled_indicies = labelled_indicies.tolist()

        if str(type(labels)) == "<class 'numpy.ndarray'>":
            if self._verbose:
                print('Converting labels to list internally')
            labels = labels.tolist()

        self.points = all_data
        self._num_points = len(self.points)
        self._num_labelled = len(labels)

        # Making a label list, with None in places where we don't have the label

        temp = [None] * self._num_points
        for i,ind in enumerate(labelled_indicies):
            temp[ind] = labels[i]
        self.labels = temp
        unlabelled_indicies = [x for x in range(self._num_points) if x not in labelled_indicies]

        # Placing each point into the correct leaf

        if self._root.is_leaf():
            self._root.labelled_index = list(labelled_indicies)
            self._root.unlabelled_index = unlabelled_indicies

        else:
            for i in labelled_indicies:
                curr_leaf = self._root.leaf_for_point(self.points[i])
                curr_leaf.labelled_index.append(i)

            for i in unlabelled_indicies:
                curr_leaf = self._root.leaf_for_point(self.points[i])
                curr_leaf.unlabelled_index.append(i)

    def label_point(self, index, value):
        '''Adds a label to a specific data point. Throws an error if that point
        is already labelled. 
        '''

        if self.labels is None:
            raise RuntimeError('No data in the tree')

        if len(self.labels) <= index:
            raise ValueError('Index {} larger than size of data in tree'.format(index))

        value = copy.copy(value)
        index = copy.copy(index)

        self.labels[index] = value
        leaf = self._root.leaf_for_point(self.points[index])
        leaf.make_labelled(index)
        self._num_labelled += 1
        self._full_leaf_mean_list_up_to_date = False
        self._full_leaf_var_list_up_to_date = False

    def add_data_point(self, new_point, label = None):
        '''Adds an additional data point to the tree, with the option of adding a label
        as well. Automatically makes it the _num_points-th point.

        Does NOT automatically grow the tree larger so you need to do that yourself.
        '''

        new_point = copy.deepcopy(new_point)
        label = copy.deepcopy(label)
        new_point = self._test_point(new_point)

        if self.points is None:
            point_index = 0
            self.points = [new_point]
            self.labels = [label]
        else:
            point_index = len(self.labels)
            self.points.append(new_point)
            self.labels.append(label)

        leaf = self._root.leaf_for_point(new_point)
        if label is None:
            leaf.unlabelled_index.append(point_index)
        else:
            leaf.labelled_index.append(point_index)

        self._full_leaf_marginal_list_up_to_date = False
        if label is not None:
            self._full_leaf_mean_list_up_to_date = False
            self._full_leaf_var_list_up_to_date = False

    ###########################################

    # Leaf list building methods: We want the tree to have a list of nodes as well as
    # various statistics about those nodes so we can easily access them. All the lists
    # will be aligned, so the ith value in a list will correspond the the ith node in the
    # node list. 

    def make_full_leaf_list(self):
        '''Makes a list with pointers to every leaf in the tree. Likely to be expensive so 
        only do this if you're pre-building a tree for extensive use later. Not needed for
        things like prediction, but needed for cell statistics and active learning.
        '''

        full_leaf_list = []
        def internal_dfs(node):
            if node.is_leaf():
                full_leaf_list.append(node)
            else:
                internal_dfs(node.left_child)
                internal_dfs(node.right_child)

        internal_dfs(self._root)
        self._full_leaf_list = full_leaf_list

        # Ensure each leaf knows where it is in the list

        for i, node in enumerate(self._full_leaf_list):
            node.full_leaf_list_pos = i
        self._full_leaf_list_up_to_date = True

    def make_full_leaf_mean_list(self):
        if not self._full_leaf_list_up_to_date:
            print('Making full leaf list. Please wait')
            self.make_full_leaf_list()
            print('Done!')

        mean_list = []
        for i, node in enumerate(self._full_leaf_list):
            label_list = [self.labels[x] for x in node.labelled_index]
            if len(label_list) != 0:
                mean_list.append(sum(label_list)/len(label_list))
            else:
                mean_list.append(0)

        self._full_leaf_mean_list = mean_list
        self._full_leaf_mean_list_up_to_date = True

    def make_full_leaf_var_list(self):
        if not self._full_leaf_list_up_to_date:
            print('Making full leaf list. Please wait')
            self.make_full_leaf_list()
            print('Done!')

        var_list = []
        for i, node in enumerate(self._full_leaf_list):
            label_list = [self.labels[x] for x in node.labelled_index]
            var_list.append(utils.unbiased_var(label_list))

        self._full_leaf_var_list = var_list
        self._full_leaf_var_list_up_to_date = True

    def make_full_leaf_marginal_list(self):
        if not self._full_leaf_list_up_to_date:
            print('Making full leaf list. Please wait')
            self.make_full_leaf_list()
            print('Done!')

        if self._num_points == 0:
            self._full_leaf_marginal_list = [0]*self._num_leaves

        else:

            marginal_list = []
            for i, node in enumerate(self._full_leaf_list):
                points_list = (
                    [self.points[x] for x in node.unlabelled_index]+
                    [self.points[x] for x in node.labelled_index])
                marginal_list.append(len(points_list)/self._num_points)

            self._full_leaf_marginal_list = marginal_list
            self._full_leaf_marginal_list_up_to_date = True

    def update_leaf_lists(self):
        self.make_full_leaf_list()
        self.make_full_leaf_mean_list()
        self.make_full_leaf_var_list()
        self.make_full_leaf_marginal_list()

    ###########################################

    # Mondrian Tree interaction methods. These methods actually use our Mondrian tree to
    # make predictions and such. 

    def predict(self, new_point):

        '''Make prediction for a data point using the mean of that leaf. If a list of points is
        given returns a list of predictions.
        '''

        try:
            if len(new_point) == 0:
                raise ValueError('No data in this new_point')
        except TypeError:
            raise TypeError(
                'Given object has no len() so it is not a point or list of '
                'points.')
        depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
        new_point_depth = depth(new_point)
        if new_point_depth > 2:
            raise ValueError('Input has too many nested structures')

        if new_point_depth == 2 or (str(type(new_point)) == "<class 'numpy.ndarray'>" and 
            len(new_point.shape) == 2):
            preds = []
            for i in range(len(new_point)):
                # print(len(new_point[i]))
                # print(new_point[i])
                preds.append(self.predict(new_point[i]))
            return preds


        new_point = copy.deepcopy(new_point)
        new_point = self._test_point(new_point)

        correct_leaf = self._root.leaf_for_point(new_point)
        if len(correct_leaf.labelled_index) == 0:
            warnings.warn(
                'WARNING: No labelled data in this leaf. The value of {} is returned by default but '
                'really should not be considered an actual prediction unless you set it with data using. '
                'the .prediction_default_value instance variable.'
                'Possible solutions to this are to sample data within that leaf, build smaller trees, '
                'or use the global data average as your prediction. But whatever solution you use dependent '
                'on what you are doing. You should be able to catch this warning and handle it automatically '
                'using the warning module with a try/except statement.'.format(self.prediction_default_value))
            return self.prediction_default_value 
        elif self._full_leaf_mean_list_up_to_date:
            return self._full_leaf_mean_list[correct_leaf.full_leaf_list_pos]

        else:
            temp_lis = [self.labels[x] for x in correct_leaf.labelled_index]
            return sum(temp_lis)/len(temp_lis)

    def get_points_in_same_leaf(self, new_point, which_index_list = 'labelled'):
        '''Gets the labelled and unlabelled point index lists for a given data point. If you want
        to predict something other than the mean (say the median), or say sample from the tree's
        predicted conditional distribution, you can use this to help implement whatever you want.

        Returns the labelled or unlabelled index lists. Labelled is the default. Pick using 
        'labelled' or 'unlabelled'
        '''

        new_point = copy.deepcopy(new_point)
        new_point = self._test_point(new_point)

        correct_leaf = self._root.leaf_for_point(new_point)
        if which_index_list == 'labelled':
            return correct_leaf.labelled_index
        elif which_index_list == 'unlabelled':
            return correct_leaf.unlabelled_index
        else:
            warnings.warn(
                'WARNING: which_index_list was not \'labelled\' or \'unlabelled\'. Defaulting to '
                'return labelled index list.')
            return correct_leaf.labelled_index

    def set_default_pred_global_mean(self):
        '''Calculates the global mean for all labelled points and sets the default prediction
        to that.
        '''

        if self.labels is None:
            self.prediction_default_value = 0
        else:
            have_labels = [x for x in self.labels if x is not None]
            if len(have_labels) != 0:
                self.prediction_default_value = sum(have_labels) / len(have_labels)
            else:
                self.prediction_default_value = 0

    ###########################################

    # Active Learning methods: methods for doing active learning as described in <paper>. all
    # methods here will start with al_ so you know they're active learning related.

    def al_set_default_var_global_var(self):
        '''Calculates the global variance for all labelled points and sets the default variance
        to that.
        '''
        if self.labels is None:
            self.al_default_var = 0
        else:
            have_labels = [x for x in self.labels if x is not None]
            if len(have_labels) != 0:
                self.al_default_var = utils.unbiased_var(have_labels)
            else:
                self.al_default_var = 0

    def al_calculate_leaf_proportions(self):
        '''Calculates estimates of the leaf proportions, using estimates for leaf variances and
        marginal probabilities, as described in <paper>
        '''

        # Ensure all the lists we need are built

        if not self._full_leaf_list_up_to_date:
            self.make_full_leaf_list()
        if not self._full_leaf_var_list_up_to_date:
            self.make_full_leaf_var_list()
        if not self._full_leaf_marginal_list_up_to_date:
            self.make_full_leaf_marginal_list()

        al_var_list = copy.copy(self._full_leaf_var_list)
        for i, val in enumerate(al_var_list):
            if val == 0:
                al_var_list[i] = self.al_default_var

        al_proportions = []

        if self._num_points == 0:
            warnings.warn('WARNING: No data points in tree. Returning uniform over all leaves')
            self._al_proportions = [1/self._num_leaves]*self._num_leaves
            self._al_proportions_up_to_date = True

        elif sum(al_var_list) == 0:
            warnings.warn('WARNING: No non-zero variance. Returning uniform over all leaves')
            self._al_proportions = [1/self._num_leaves]*self._num_leaves
            self._al_proportions_up_to_date = True

        else:
            for i, node in enumerate(self._full_leaf_list):
                al_proportions.append(math.sqrt(
                    self._full_leaf_marginal_list[i] * al_var_list[i]))

            normalizer = sum(al_proportions)
            al_proportions = [x/normalizer for x in al_proportions]
            self._al_proportions = al_proportions
            self._al_proportions_up_to_date = True

    def al_calculate_leaf_number_new_labels(self, num_samples_total, round_by = 'smallest'):
        '''Calculate how many new labelled points each leaf should get to get as close as 
        possible to the proportions in _al_proportions. Since these proportions might not
        be possible with integer number of points, we have two heuristic ways of making
        integer if we have too few point. 

        The first is to floor every number and then add one to the leaves with the highest
        fractions until we've hit our budget.

        The second is to floor every number and then add one to the leaves with the smallest
        number of points. This is the current default.

        If we have too many points, we remove from the largest leaves till we are back in our 
        budget.
        '''
        num_samples_total = copy.copy(num_samples_total)

        if num_samples_total < self._num_labelled:
            raise ValueError('The total given number of samples has already been exceeded.')

        if num_samples_total > self._num_points:
            raise ValueError('The total number of samples is greater than the number of points.')

        if not self._al_proportions_up_to_date:
            print('Calculating leaf proportions. Please wait')
            self.al_calculate_leaf_proportions()
            print('Done!')

        num_samples_left = num_samples_total - self._num_labelled

        # Calculate the optimal (fractional) number of points per leaf

        num_per_leaf_fractions = [x*num_samples_total for x in self._al_proportions]

        # Calculate the number of labelled points that should be added to approach that optimal

        current_num_per_leaf = []
        for i, node in enumerate(self._full_leaf_list):
            current_num_per_leaf.append(len(node.labelled_index))

        num_per_leaf = [max(0,math.floor(x) - current_num_per_leaf[i]) for i, x in enumerate(
            num_per_leaf_fractions)]

        remaining_budget = num_samples_left - sum(num_per_leaf)
        if abs(remaining_budget/num_samples_left) > 0.1:
            warnings.warn('remaining_budget is = {} fraction of number of new samples. '
            'It may not be possible to get close to the optimal solution given the current locations '
            'of labelled data.'.format(abs(remaining_budget/num_samples_left))
            )

        # print(remaining_budget)

        # If we too few points we use one of two heuristics

        if round_by == 'highest':
            num_per_leaf_fractions = [x - math.floor(x) for x in num_per_leaf_fractions]
            while remaining_budget > 0:
                num_per_leaf[num_per_leaf_fractions.index(max(num_per_leaf_fractions))] += 1
                num_per_leaf_fractions[num_per_leaf_fractions.index(max(num_per_leaf_fractions))] = 0
                remaining_budget -= 1

                # If we've added one to every leaf, start adding again to highest fraction leaves.

                if all([x==0 for x in num_per_leaf_fractions]):
                    num_per_leaf_fractions = [x - math.floor(x) for x in num_per_leaf_fractions]

        elif round_by == 'smallest':
            total_num_per_leaf = [math.floor(x*num_samples_total) for x in self._al_proportions]
            while remaining_budget > 0:
                num_per_leaf[total_num_per_leaf.index(min(total_num_per_leaf))] += 1
                total_num_per_leaf[total_num_per_leaf.index(min(total_num_per_leaf))] = float('inf')
                remaining_budget -= 1

                # If we've added one to every leaf, start adding again to smallest leaves.

                if all([math.isinf(x) for x in total_num_per_leaf]):
                    total_num_per_leaf = [math.floor(x) for x in self._al_proportions]

        else:
            raise ValueError('Invalid round_by')

        # If we have too many points, we subtract from the leaves with the most total points
        # under the optimal solution. This occurs when leaves have already exceeded their
        # optimal sampling number during the random sampling phase.

        total_num_per_leaf = [math.floor(x*num_samples_total) for x in self._al_proportions]
        while remaining_budget < 0:

            for i, val in enumerate(num_per_leaf):
                if val == 0:
                    total_num_per_leaf[i] = float('-inf')

            num_per_leaf[total_num_per_leaf.index(max(total_num_per_leaf))] -= 1
            total_num_per_leaf[total_num_per_leaf.index(max(total_num_per_leaf))] -=1
            remaining_budget += 1

        self._al_leaf_number_new_labels = num_per_leaf

    def al_calculate_point_probabilities_proportions(self):
        '''Calculate the corresponding probabilities given to each point in order to achieve
        the correct leaf proportions. Each point is given weight of the leaf divided by the
        number of unlabelled points in the leaf.
        '''

        if not self._al_proportions_up_to_date:
            print('Calculating leaf proportions. Please wait')
            self.al_calculate_leaf_proportions()
            print('Done!')

        point_prob_list = [None] * self._num_points
        for i, node in enumerate(self._full_leaf_list):
            for ind in node.unlabelled_index:
                point_prob_list[ind] = self._al_proportions[i] / len(node.unlabelled_index)

        self._al_point_weights_proportional = point_prob_list

    def al_calculate_point_probabilities_adjustment(self, num_samples_total):
        '''Calculate the corresponding probabilities given to each point such that in expectation
        the number of points sampled from each leaf will be the leaf proportion times the 
        num_samples_total.

        If the leaf has already had more samples than expected, gives probability 0. All 
        probabilities are normalized to account for rounding issues and passive oversampling of
        leaves (NOTE: AD HOC SOLUTION TO PROBLEM. MAKE SURE IT MAKES SENSE)
        '''

        num_samples_total = copy.copy(num_samples_total)

        if not self._al_proportions_up_to_date:
            print('Calculating leaf proportions. Please wait')
            self.al_calculate_leaf_proportions()
            print('Done!')

        num_samples_left = num_samples_total - self._num_labelled

        point_prob_list = [None] * self._num_points
        for i, node in enumerate(self._full_leaf_list):
            num_already = len(node.labelled_index)
            num_expected = num_samples_total * self._al_proportions[i]
            for ind in node.unlabelled_index:
                point_prob_list[ind] = max(0,
                    (num_expected - num_already)/(num_samples_left*len(node.unlabelled_index)))

        # print(point_prob_list)
        tot = sum([x for x in point_prob_list if x is not None])
        if tot != 0:
            for i, val in enumerate(point_prob_list):
                if val is not None:
                    point_prob_list[i] = point_prob_list[i]/tot

        self._al_point_weights_adjustment = point_prob_list

