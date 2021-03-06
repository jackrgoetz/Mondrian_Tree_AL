from Mondrian_Tree import Mondrian_Tree
import warnings
import numpy as np

def scale_zero_one(col):

    offset = min(col)
    scale = max(col) - min(col)

    col = (col - offset)/scale
    return(col)

data = np.loadtxt('data_sets/datacl_scaled.csv', delimiter=',')

X = data[:,:-1]
for i in range(X.shape[1]):
    X[:,i] = scale_zero_one(X[:,i])

y = data[:,-1]

train_test_seed = 1
np.random.seed(train_test_seed)

cv_ind = np.random.permutation(range(X.shape[0]))

train_ind = cv_ind[:1000]
test_ind = cv_ind[1000:]

X_train = X[train_ind,:]
X_test = X[test_ind,:]

y_train = y[train_ind]
y_test = y[test_ind]

n, p = X_train.shape
# print(n,p)

seed = 14

MT = Mondrian_Tree([[0,1]]*p)
MT.update_life_time(X.shape[0]**(1/(2+p))-1, set_seed=seed)
# print(MT._num_leaves)
MT.input_data(np.concatenate((X_train, X_test),axis=0), range(1000), y_train)

MT.update_leaf_lists()
used_leaf_counter = 0
for node in MT._full_leaf_list:
    if len(node.labelled_index) != 0:
        # print(len(node.labelled_index), node.calculate_cell_l2_diameter())
        # print('var = {}'.format(MT._full_leaf_var_list[node.full_leaf_list_pos]))
        used_leaf_counter += 1
print('number of used leaves = {}'.format(used_leaf_counter))
MT.al_set_default_var_global_var()
# print(MT.al_default_var)

MT.al_calculate_leaf_proportions()
MT.al_calculate_leaf_number_new_labels(1500)

for i, node in enumerate(MT._full_leaf_list):
    curr_num = len(node.labelled_index)
    tot_num = curr_num + MT._al_leaf_number_new_labels[i]
    print(curr_num,tot_num, MT._al_proportions[i] * 1500,MT._al_leaf_number_new_labels[i])

# MT.set_default_pred_global_mean()

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     MT_preds = MT.predict(X_test)
# MT_preds = np.array(MT_preds)

# print(sum(1/n*(y_test - MT_preds)**2))