from Breiman_Tree import Breiman_Tree
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy

# n_finals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_finals = [100, 200, 300, 400, 500]
batch_size = 1
min_samples_leaf = 5

# n_finals = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# p = 5

data_seeds = [x * 11 for x in range(10)]
tree_seeds = [x * 13 for x in range(10)]

BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])
BT_uc_MSE = np.zeros([len(n_finals)])

def scale_zero_one(col):

    offset = min(col)
    scale = max(col) - min(col)

    col = (col - offset)/scale
    return(col)

ccpp_data = np.genfromtxt('data_sets/datacl_scaled.csv', delimiter = ',')

X = ccpp_data[:,:-1]
for i in range(X.shape[1]):
    X[:,i] = scale_zero_one(X[:,i])

y = ccpp_data[:,-1]

n,p = X.shape

for n_final_ind, n_final in enumerate(n_finals):

    n_start = int(n_final/2)

    for data_seed in data_seeds:

        np.random.seed(data_seed)

        cv_ind = np.random.permutation(range(X.shape[0]))

        train_ind_al = cv_ind[:n_start]
        train_ind_rn = cv_ind[:n_final]

        X = X[cv_ind,:]
        y = y[cv_ind]

        X_test = X[cv_ind[n_start:],:]
        y_test = y[cv_ind[n_start:]]

        for tree_seed in tree_seeds:

            print(n_final, data_seed, tree_seed)

            # BT_al

            BT_al = Breiman_Tree(seed=tree_seed, min_samples_leaf=min_samples_leaf)
            BT_al.input_data(X, range(n_start), y[:n_start])
            for _ in range(int((n_final - n_start)/batch_size)):
                # print(_)
                BT_al.fit_tree()
                BT_al.al_calculate_leaf_proportions()
                new_points = BT_al.pick_new_points(num_samples = batch_size)
                for new_point in new_points:
                    BT_al.label_point(new_point, y[new_point])

            BT_al.fit_tree()
            BT_al_preds = BT_al.predict(X_test)
            BT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_al_preds)**2)
            # print('Done BT_al')

            # BT_rn

            BT_rn = DecisionTreeRegressor(random_state=tree_seed, min_samples_leaf=min_samples_leaf)
            BT_rn.fit(X[list(range(n_final)),:], y[list(range(n_final))])
            BT_rn_preds = BT_rn.predict(X_test)
            BT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_rn_preds)**2)
            # print('Done BT_rn')

            # BT_uc
            BT_uc = Breiman_Tree(seed = tree_seed, min_samples_leaf=min_samples_leaf)
            BT_uc.input_data(X, range(n_start), y[:n_start])
            for _ in range(int((n_final - n_start)/batch_size)):
                # print(_)
                BT_uc.fit_tree()
                BT_uc.calculate_leaf_statistics()
                BT_uc._al_proportions = np.array(BT_uc._leaf_var)/sum(BT_uc._leaf_var)
                BT_uc._leaf_proportions_up_to_date = True
                new_points = BT_uc.pick_new_points(num_samples = batch_size)
                for new_point in new_points:
                    BT_uc.label_point(new_point, y[new_point])

            BT_uc.fit_tree()
            BT_uc_preds = BT_uc.predict(X_test)
            BT_uc_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_uc_preds)**2)

BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))
BT_uc_MSE = BT_uc_MSE/(len(data_seeds) * len(tree_seeds))

np.savez('graphs/sim_cl_BT_' + str(min_samples_leaf) + '_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.npz', 
    BT_uc_MSE=BT_uc_MSE,
    BT_al_MSE=BT_al_MSE, BT_rn_MSE=BT_rn_MSE)

f, axarr = plt.subplots(2, sharex=True)

# mt_al = axarr[0].plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active sampling')
# mt_rn = axarr[0].plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random sampling')
# mt_uc = axarr[0].plot(n_finals, MT_uc_MSE, color = 'green', label = 'Mondrian Tree - Uncertainty sampling')
# mt_oracle = axarr[0].plot(n_finals, MT_oracle_MSE, color = 'black', label='Oracle Mondrian Tree')
# axarr[0].set_title('Heteroskedastic simulation')
# axarr[0].legend(loc='best')

bt_al = axarr[1].plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Breiman Tree - Active sampling')
bt_rn = axarr[1].plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Breiman Tree - Random sampling')
bt_rn = axarr[1].plot(n_finals, BT_uc_MSE, color = 'green', linestyle = '--',
    label = 'Breiman Tree - Uncertainty sampling')
axarr[1].legend(loc='best')

f.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
f.text(0.5, 0.01, 'Final number of labelled points', ha='center')

plt.tight_layout()
plt.savefig('graphs/sim_cl_BT_' + str(min_samples_leaf) + '_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.pdf')