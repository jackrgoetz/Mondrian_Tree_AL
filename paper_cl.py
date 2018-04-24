from Mondrian_Tree import Mondrian_Tree
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import warnings
import matplotlib.pyplot as plt

n_finals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
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

            MT_al = Mondrian_Tree([[0,1]]*p)
            MT_al.update_life_time(n_final**(1/(2+p))-1, set_seed=tree_seed)
            # print(MT_al._num_leaves)
            MT_al.input_data(X, range(n_start), y[:n_start])

            MT_al.make_full_leaf_list()
            MT_al.make_full_leaf_var_list()
            MT_al.al_set_default_var_global_var()
            # print(MT_al.al_default_var)

            MT_al.al_calculate_leaf_proportions()
            MT_al.al_calculate_leaf_number_new_labels(n_final)

            new_labelled_points = []
            for i, node in enumerate(MT_al._full_leaf_list):
                # print(i)
                curr_num = len(node.labelled_index)
                tot_num = curr_num + MT_al._al_leaf_number_new_labels[i]
                # print(curr_num,tot_num, MT_al._al_proportions[i] * n_final,node.rounded_linear_dims(2))
                num_new_points = MT_al._al_leaf_number_new_labels[i]
                labels_to_add = node.pick_new_points(num_new_points,self_update = False, set_seed = tree_seed*i)
                # print(labels_to_add)
                new_labelled_points.extend(labels_to_add)
                for ind in labels_to_add:
                    MT_al.label_point(ind, y[ind])

            MT_al.set_default_pred_global_mean()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_al_preds = MT_al.predict(X_test)
            MT_al_preds = np.array(MT_al_preds)
            MT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_al_preds)**2)

            # print('Done MT_al')

            # MT_rn

            MT_rn = Mondrian_Tree([[0,1]]*p)
            MT_rn.update_life_time(n_final**(1/(2+p))-1, set_seed=tree_seed)
            # print(MT._num_leaves)
            MT_rn.input_data(X, range(n_final), y[:n_final])
            MT_rn.set_default_pred_global_mean()
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

            BT_al = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = MT_al._num_leaves)
            BT_al.fit(X[list(range(n_start)) + new_labelled_points,:], y[list(range(n_start)) + new_labelled_points])
            BT_al_preds = BT_al.predict(X_test)
            BT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_al_preds)**2)
            # print('Done BT_al')

            BT_rn = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = MT_rn._num_leaves)
            BT_rn.fit(X[list(range(n_final)),:], y[list(range(n_final))])
            BT_rn_preds = BT_rn.predict(X_test)
            BT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_rn_preds)**2)
            # print('Done BT_rn')

MT_al_MSE = MT_al_MSE/(len(data_seeds) * len(tree_seeds))
MT_rn_MSE = MT_rn_MSE/(len(data_seeds) * len(tree_seeds))
MT_oracle_MSE = MT_oracle_MSE/(len(data_seeds) * len(tree_seeds))
BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))

plt.plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active labelling')
plt.plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random labelling')

plt.title('Cl experiment')
plt.xlabel('Final number of labelled points')
plt.ylabel('MSE')
# plt.show()

# plt.clf()

plt.plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Breiman Tree - Active labelling')
plt.plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Breiman Tree - Random labelling')
plt.legend(loc="best")
plt.savefig('graphs/cl.pdf')
plt.show()