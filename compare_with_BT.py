from sklearn.tree import DecisionTreeRegressor
from Mondrian_Tree import Mondrian_Tree

import numpy as np
import matplotlib.pyplot as plt

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

cv_ind = np.random.permutation(range(X.shape[0]))

train_ind = cv_ind[:1000]
test_ind = cv_ind[1000:]

X_train = X[train_ind,:]
X_test = X[test_ind,:]

y_train = y[train_ind]
y_test = y[test_ind]

n, p = X_train.shape
print(n,p)

seed = 4

MT = Mondrian_Tree([[0,1]]*p)
MT.update_life_time(n**(1/(2+p))-1, set_seed=seed)
# print(MT._num_leaves)
MT.input_data(X_train, range(1000), y_train)

MT.make_full_leaf_list()
used_leaf_counter = 0
for node in MT._full_leaf_list:
    if len(node.labelled_index) != 0:
        print(len(node.labelled_index), node.calculate_cell_l2_diameter())
        used_leaf_counter += 1
print(used_leaf_counter)

MT.set_default_pred_global_mean()
MT_preds = MT.predict(X_test)
MT_preds = np.array(MT_preds)

print(sum(1/n*(y_test - MT_preds)**2))

BT = DecisionTreeRegressor(random_state=seed, max_leaf_nodes = used_leaf_counter)
BT.fit(X_train, y_train)
BT_preds = BT.predict(X_test)

print(sum(1/n*(y_test - BT_preds)**2))