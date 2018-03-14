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

n, p = X.shape
print(n,p)

seed = 1

MT = Mondrian_Tree([[0,1]]*p)
MT.update_life_time(n**(1/(2+p))-1, set_seed=seed)
print(MT._num_leaves)
MT.input_data(X, range(n), y)

MT_preds = MT.predict(X)
MT_preds = np.array(MT_preds)

print(sum(1/n*(y - MT_preds)**2))

BT = DecisionTreeRegressor(random_state=seed, max_leaf_nodes = MT._num_leaves)
BT.fit(X, y)
BT_preds = BT.predict(X)

print(sum(1/n*(y - BT_preds)**2))