from data_sets.toy_data_pw_linear import toy_data_pw_linear
from Mondrian_Tree import Mondrian_Tree
import numpy as np
import warnings
import itertools

n_points = 100000
n_start = 500
n_final = 1000
p = 2

low_constant = 1
high_constant = 100
std = 2

high_area = [[0.75,1],[0.75,1]]

data_seed = 10

X, y = toy_data_pw_linear(n=n_points,p=p,high_area=high_area,high_constant=high_constant,
    low_constant=low_constant,std=std, set_seed=data_seed)

X = np.array(X)
y = np.array(y)

train_test_seed = 1
np.random.seed(train_test_seed)

cv_ind = np.random.permutation(range(X.shape[0]))

train_ind_al = cv_ind[:n_start]
train_ind_rn = cv_ind[:n_final]
test_ind = cv_ind[n_start:]

X_train = X[train_ind_al,:]
# X_train_rn = X[train_ind_rn,:]
X_test = X[test_ind,:]

y_train_al = y[train_ind_al]
y_train_rn = y[train_ind_rn]
y_test = y[test_ind]

X = X[cv_ind,:]
y = y[cv_ind]

# print(y)

n, p = X_train.shape
# print(n,p)

seed = 13

MT = Mondrian_Tree([[0,1]]*p)
MT.update_life_time(n_final**(1/(2+p))-1, set_seed=seed)
# print(MT._num_leaves)
MT.input_data(np.concatenate((X_train, X_test),axis=0), range(n_start), y_train_al)

MT.make_full_leaf_list()
MT.make_full_leaf_var_list()
used_leaf_counter = 0
for node in MT._full_leaf_list:
    if len(node.labelled_index) != 0:
        # print(len(node.labelled_index), node.calculate_cell_l2_diameter())
        # print('var = {}'.format(MT._full_leaf_var_list[node.full_leaf_list_pos]))
        used_leaf_counter += 1
print('number of used leaves = {}'.format(used_leaf_counter))
MT.al_set_default_var_global_var()
print(MT.al_default_var)

MT.al_calculate_leaf_proportions()
MT.al_calculate_leaf_number_new_labels(n_final)

for i, node in enumerate(MT._full_leaf_list):
    # print(i)
    curr_num = len(node.labelled_index)
    tot_num = curr_num + MT._al_leaf_number_new_labels[i]
    print(curr_num,tot_num, MT._al_proportions[i] * n_final,node.rounded_linear_dims(2))
    num_new_points = MT._al_leaf_number_new_labels[i]
    labels_to_add = node.pick_new_points(num_new_points,self_update = False, set_seed = seed*i)
    # print(labels_to_add)
    for ind in labels_to_add:
        MT.label_point(ind, y[ind])

MT.set_default_pred_global_mean()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    MT_preds = MT.predict(X_test)
MT_preds = np.array(MT_preds)

print('MSE from AL = {}'.format(sum(1/X_test.shape[0]*(y_test - MT_preds)**2)))

MT2 = Mondrian_Tree([[0,1]]*p)
MT2.update_life_time(n_final**(1/(2+p))-1, set_seed=seed)
# print(MT._num_leaves)
MT2.input_data(np.concatenate((X_train, X_test),axis=0), range(n_final), y_train_rn)
MT2.set_default_pred_global_mean()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    MT2_preds = MT2.predict(X_test)
MT2_preds = np.array(MT2_preds)
# print(MT2_preds)
print('MSE from random = {}'.format(sum(1/X_test.shape[0]*(y_test - MT2_preds)**2)))

# Building the oracle tree, i.e. a tree with a grid of points and noiseless responses to get the
# oracle MSE. Only viable for p = 2

granularity = 0.01
xx, yy = np.meshgrid(np.arange(0,1,granularity), np.arange(0,1,granularity))
point_grid = list(zip(xx.flatten(), yy.flatten()))
function_vals = []

for point in point_grid:
    is_high_slope = []
    for j,val in enumerate(point):
        if val > high_area[j][0] and val < high_area[j][1]:
            is_high_slope.append(True)
        else:
            is_high_slope.append(False)

    if all(is_high_slope):
        function_vals.append(high_constant*sum(point))
    else:
        function_vals.append(low_constant*sum(point))
# print(len(function_vals))

MT_oracle = Mondrian_Tree([[0,1]]*p)
MT_oracle.update_life_time(n_final**(1/(2+p))-1, set_seed=seed)
MT_oracle.input_data(point_grid, range(len(function_vals)), function_vals)
oracle_preds = MT_oracle.predict(X_test)

print('MSE from oracle = {}'.format(sum(1/X_test.shape[0]*(y_test - oracle_preds)**2)))