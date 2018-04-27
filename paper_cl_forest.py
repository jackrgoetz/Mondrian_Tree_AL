from Mondrian_Forest import Mondrian_Forest
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import warnings
import matplotlib.pyplot as plt

n_finals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_tree = 50
# n_finals = [2000]

data_seeds = [x * 11 for x in range(5)]
tree_seeds = [x * 13 for x in range(5)]

MT_al_MSE = np.zeros([len(n_finals)])
MT_rn_MSE = np.zeros([len(n_finals)])
MT_oracle_MSE = np.zeros([len(n_finals)])
BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])

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

print(n, p)

for n_final_ind, n_final in enumerate(n_finals):

    n_start = int(n_final/2)

    for data_seed in data_seeds:

        # plt.scatter(X[:,0], X[:,1], c=y)
        # plt.show()
        # sys.exit()

        np.random.seed(data_seed)

        cv_ind = np.random.permutation(range(X.shape[0]))

        train_ind_al = cv_ind[:n_start]
        train_ind_rn = cv_ind[:n_final]

        X = X[cv_ind,:]
        y = y[cv_ind]

        X_test = X[cv_ind[n_start:],:]
        y_test = y[cv_ind[n_start:]]

        # test_ind = cv_ind[n_start:]

        # X_train = X[train_ind_al,:]
        # # X_train_rn = X[train_ind_rn,:]
        # X_test = X[test_ind,:]

        # y_train_al = y[train_ind_al]
        # y_train_rn = y[train_ind_rn]
        # y_test = y[test_ind]

        # X = X[cv_ind,:]
        # y = y[cv_ind] 

        for tree_seed in tree_seeds:

            print(n_final, data_seed, tree_seed)

            # MT_al and labels for BT_al

            MT_al = Mondrian_Forest([[0,1]]*p, n_tree)
            MT_al.update_life_time(n_final**(1/(2+p))-1, 
                set_seeds=[n_tree*tree_seed + x for x in range(n_tree)])
            # print(MT_al._num_leaves)
            MT_al.input_data(X, range(n_start), y[:n_start])

            MT_al.al_average_point_probabilities_adjustment(n_final)

            new_labelled_points = list(np.random.choice(list(range(n)), 
                p = MT_al._al_avg_weights_adjustment, size=n_start, replace = False))
            for ind in new_labelled_points:
                MT_al.label_point(ind, y[ind])

            # print(len(new_labelled_points))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_al_preds = MT_al.predict(X_test)
            MT_al_preds = np.array(MT_al_preds)
            MT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_al_preds)**2)

            # print('Done MT_al')

            # MT_rn

            MT_rn = Mondrian_Forest([[0,1]]*p, n_tree)
            MT_rn.update_life_time(n_final**(1/(2+p))-1, 
                set_seeds=[n_tree*tree_seed + x for x in range(n_tree)])
            # print(MT._num_leaves)
            MT_rn.input_data(X, range(n_final), y[:n_final])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_rn_preds = MT_rn.predict(X_test)
            MT_rn_preds = np.array(MT_rn_preds)
            MT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_rn_preds)**2)

            # print('Done MT_rn')

            # MT_oracle - meaningless here

            MT_oracle_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test)**2)

            # print('Done MT_oracle')

            # BT_al

            BT_al = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = int(MT_al._avg_num_leaves))
            BT_al.fit(X[list(range(n_start)) + new_labelled_points,:], y[list(range(n_start)) + new_labelled_points])
            BT_al_preds = BT_al.predict(X_test)
            BT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_al_preds)**2)
            # print('Done BT_al')

            BT_rn = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = int(MT_rn._avg_num_leaves))
            BT_rn.fit(X[list(range(n_final)),:], y[list(range(n_final))])
            BT_rn_preds = BT_rn.predict(X_test)
            BT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_rn_preds)**2)
            # print('Done BT_rn')
            # print('Done BT_rn')

MT_al_MSE = MT_al_MSE/(len(data_seeds) * len(tree_seeds))
MT_rn_MSE = MT_rn_MSE/(len(data_seeds) * len(tree_seeds))
MT_oracle_MSE = MT_oracle_MSE/(len(data_seeds) * len(tree_seeds))
BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))

f, axarr = plt.subplots(2, sharex=True)

mt_al = axarr[0].plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active labelling')
mt_rn = axarr[0].plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random labelling')
axarr[0].set_title('Cl simulation')
# axarr[0].legend(loc='best')

bt_al = axarr[1].plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Breiman Tree - Active labelling')
bt_rn = axarr[1].plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Breiman Tree - Random labelling')
# axarr[1].legend(loc='best')

f.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
f.text(0.5, 0.01, 'Final number of labelled points', ha='center')

plt.tight_layout()
plt.savefig('graphs/sim_cl_forest.pdf')
plt.show()

# plt.plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active labelling')
# plt.plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random labelling')

# plt.title('Cl experiment')
# plt.xlabel('Final number of labelled points')
# plt.ylabel('MSE')
# # plt.show()

# # plt.clf()

# plt.plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
#     label = 'Breiman Tree - Active labelling')
# plt.plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
#     label = 'Breiman Tree - Random labelling')
# plt.legend(loc="best")
# plt.savefig('graphs/cl.pdf')
# plt.show()