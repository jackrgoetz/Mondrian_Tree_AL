from data_sets.toy_data_var import toy_data_var
from Mondrian_Forest import Mondrian_Forest
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy
import math

n_points = 40000
n_test_points = 5000
n_finals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
p = 10
marginal = 'uniform'
n_tree = 2
# n_finals = [2000]

data_seeds = [x * 11 for x in range(1)]
tree_seeds = [x * 13 for x in range(1)]

constant = 0
low_std = 1
high_std = 5

high_area = [[0.5,1]]*p

MT_al_MSE = np.zeros([len(n_finals)])
MT_rn_MSE = np.zeros([len(n_finals)])
MT_uc_MSE = np.zeros([len(n_finals)])

BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])
BT_uc_MSE = np.zeros([len(n_finals)])

for n_final_ind, n_final in enumerate(n_finals):

    n_start = int(n_final/2)

    for data_seed in data_seeds:

        # plt.scatter(X[:,0], X[:,1], c=y)
        # plt.show()
        # sys.exit()

        X, y = toy_data_var(n=n_points,p=p,high_area=high_area,constant=constant,
            low_std=low_std,high_std=high_std, set_seed=data_seed, marginal=marginal)

        X = np.array(X)
        y = np.array(y)

        np.random.seed(data_seed)

        cv_ind = np.random.permutation(range(X.shape[0]))

        train_ind_al = cv_ind[:n_start]
        train_ind_rn = cv_ind[:n_final]

        X = X[cv_ind,:]
        y = y[cv_ind]

        X_test, y_test = toy_data_var(n=n_test_points,p=p,high_area=high_area,constant=constant,
            low_std=low_std,high_std=high_std, set_seed=data_seed+1,marginal=marginal)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        for tree_seed in tree_seeds:

            print(n_final, data_seed, tree_seed)

            # MT_al and labels for BT_al

            MT_al = Mondrian_Forest([[0,1]]*p, n_tree)
            MT_al.update_life_time(n_final**(1/(2+p))-1, 
                set_seeds=[n_tree*tree_seed + x for x in range(n_tree)])
            MT_rn = copy.deepcopy(MT_al)

            # print(MT_al._num_leaves)
            MT_al.input_data(X, range(n_start), y[:n_start])

            MT_al.al_average_point_probabilities_adjustment(n_final)

            MT_uc = copy.deepcopy(MT_al)

            new_labelled_points = list(np.random.choice(list(range(n)), 
                p = MT_al._al_avg_weights_adjustment, size=n_final - n_start, replace = False))
            for ind in new_labelled_points:
                MT_al.label_point(ind, y[ind])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_al_preds = MT_al.predict(X_test)
            MT_al_preds = np.array(MT_al_preds)
            MT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_al_preds)**2)

            # print('Done MT_al')

            # MT_rn

            MT_rn.input_data(X, range(n_final), y[:n_final])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_rn_preds = MT_rn.predict(X_test)
            MT_rn_preds = np.array(MT_rn_preds)
            MT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_rn_preds)**2)

            # print('Done MT_rn')

            # MT_uc

            new_labelled_points_uc = []
            probs = np.zeros([n])
            for tree in MT_uc.tree_list:
                tree.al_set_default_var_global_var()
                for ind, val in enumerate(tree._full_leaf_var_list):
                    if val == 0:
                        tree._full_leaf_var_list[ind] = tree.al_default_var
                tree._al_proportions = [x / sum(tree._full_leaf_var_list) for x in tree._full_leaf_var_list]
                tree._al_proportions_up_to_date = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree.al_calculate_leaf_number_new_labels(n_final)
                    tree._al_proportions = [x / sum(tree._al_leaf_number_new_labels) for x in tree._al_leaf_number_new_labels]
                    tree.al_calculate_point_probabilities_adjustment(n_final)
                    for ind, val in enumerate(tree._al_point_weights_adjustment):
                        if val is None:
                            tree._al_point_weights_adjustment[ind] = 0

                probs = probs + np.array(tree._al_point_weights_adjustment)

            probs = probs / sum(probs)

            new_labelled_points_uc = list(np.random.choice(list(range(n)), 
                p = probs, size=n_final - n_start, replace = False))
            for ind in new_labelled_points_uc:
                MT_uc.label_point(ind, y[ind])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_uc_preds = MT_uc.predict(X_test)
            MT_uc_preds = np.array(MT_uc_preds)
            MT_uc_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_uc_preds)**2)


            # BT_al

            BT_al = RandomForestRegressor(n_estimators = n_tree, max_features = 1./3, 
                random_state=tree_seed, max_leaf_nodes = math.ceil(MT_al._avg_num_leaves)+1)
            BT_al.fit(X[list(range(n_start)) + new_labelled_points,:], y[list(range(n_start)) + new_labelled_points])
            BT_al_preds = BT_al.predict(X_test)
            BT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_al_preds)**2)
            # print('Done BT_al')

            # BT_rn

            BT_rn = RandomForestRegressor(n_estimators = n_tree, max_features = 1./3, 
                random_state=tree_seed, max_leaf_nodes = math.ceil(MT_al._avg_num_leaves)+1)
            BT_rn.fit(X[list(range(n_final)),:], y[list(range(n_final))])
            BT_rn_preds = BT_rn.predict(X_test)
            BT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_rn_preds)**2)
            # print('Done BT_rn')

            # BT_uc

            BT_uc = RandomForestRegressor(n_estimators = n_tree, max_features = 1./3, 
                random_state=tree_seed, max_leaf_nodes = math.ceil(MT_uc._avg_num_leaves)+1)
            BT_uc.fit(X[list(range(n_start)) + new_labelled_points_uc,:], y[list(range(n_start)) + new_labelled_points_uc])
            BT_uc_preds = BT_uc.predict(X_test)
            BT_uc_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_uc_preds)**2)



MT_al_MSE = MT_al_MSE/(len(data_seeds) * len(tree_seeds))
MT_rn_MSE = MT_rn_MSE/(len(data_seeds) * len(tree_seeds))
MT_uc_MSE = MT_uc_MSE/(len(data_seeds) * len(tree_seeds))

BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))
BT_uc_MSE = BT_uc_MSE/(len(data_seeds) * len(tree_seeds))

np.savez('graphs/sim_heteroskedastic_forest_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.npz', 
    MT_al_MSE=MT_al_MSE, MT_rn_MSE=MT_rn_MSE, 
    MT_uc_MSE=MT_uc_MSE, BT_uc_MSE=BT_uc_MSE,
    BT_al_MSE=BT_al_MSE, BT_rn_MSE=BT_rn_MSE)

f, axarr = plt.subplots(2, sharex=True)

mt_al = axarr[0].plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Forest - Active sampling')
mt_rn = axarr[0].plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Forest - Random sampling')
mt_uc = axarr[0].plot(n_finals, MT_uc_MSE, color = 'green', label = 'Mondrian Forest - Uncertainty sampling')
axarr[0].set_title('Cl experiment')
axarr[0].legend(loc='best')

bt_al = axarr[1].plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Random Forest - Active sampling')
bt_rn = axarr[1].plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Random Forest - Random sampling')
bt_rn = axarr[1].plot(n_finals, BT_uc_MSE, color = 'green', linestyle = '--',
    label = 'Random Forest - Uncertainty sampling')
axarr[1].legend(loc='best')

f.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
f.text(0.5, 0.01, 'Final number of labelled points', ha='center')

plt.tight_layout()
plt.savefig('graphs/sim_heteroskedastic_forest_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.pdf')