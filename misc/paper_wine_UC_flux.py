from Mondrian_Tree import Mondrian_Tree
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy

n_finals = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# n_finals = [2000]

data_seeds = [x * 11 for x in range(5)]
tree_seeds = [x * 13 for x in range(5)]

MT_al_MSE = np.zeros([len(n_finals)])
MT_rn_MSE = np.zeros([len(n_finals)])
MT_uc_MSE = np.zeros([len(n_finals)])

BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])
BT_uc_MSE = np.zeros([len(n_finals)])

def scale_zero_one(col):

    offset = min(col)
    scale = max(col) - min(col)

    col = (col - offset)/scale
    return(col)

ccpp_data = np.genfromtxt('data_sets/winequality_white.csv', delimiter = ',')

X = ccpp_data[:,:-1]
for i in range(X.shape[1]):
    X[:,i] = scale_zero_one(X[:,i])

y = ccpp_data[:,-1]

n,p = X.shape

print(n, p)

for n_final_ind, n_final in enumerate(n_finals):

    # Saving results during run

    if n_final_ind > 1:

        MT_al_MSE_temp = MT_al_MSE/(len(data_seeds) * len(tree_seeds))
        MT_rn_MSE_temp = MT_rn_MSE/(len(data_seeds) * len(tree_seeds))
        MT_uc_MSE_temp = MT_uc_MSE/(len(data_seeds) * len(tree_seeds))

        BT_al_MSE_temp = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
        BT_rn_MSE_temp = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))
        BT_uc_MSE_temp = BT_uc_MSE/(len(data_seeds) * len(tree_seeds))

        np.savez('graphs/sim_wine_uc_' + 
            str(len(data_seeds) * len(tree_seeds)) + '.npz', 
            MT_al_MSE=MT_al_MSE_temp, MT_rn_MSE=MT_rn_MSE_temp, 
            MT_uc_MSE=MT_uc_MSE_temp, BT_uc_MSE=BT_uc_MSE_temp,
            BT_al_MSE=BT_al_MSE_temp, BT_rn_MSE=BT_rn_MSE_temp)

        f, axarr = plt.subplots(2, sharex=True)

        mt_al = axarr[0].plot(n_finals[:n_final_ind], MT_al_MSE_temp[:n_final_ind], 
            color = 'red', label='Mondrian Tree - Active sampling')
        mt_rn = axarr[0].plot(n_finals[:n_final_ind], MT_rn_MSE_temp[:n_final_ind], 
            color = 'blue', label = 'Mondrian Tree - Random sampling')
        mt_uc = axarr[0].plot(n_finals[:n_final_ind], MT_uc_MSE_temp[:n_final_ind], 
            color = 'green', label = 'Mondrian Tree - Uncertainty sampling')
        axarr[0].set_title('Wine experiment')
        axarr[0].legend(loc='best')

        bt_al = axarr[1].plot(n_finals[:n_final_ind], BT_al_MSE_temp[:n_final_ind], 
            color = 'red', linestyle = '--', label = 'Breiman Tree - Active sampling')
        bt_rn = axarr[1].plot(n_finals[:n_final_ind], BT_rn_MSE_temp[:n_final_ind], 
            color = 'blue', linestyle = '--',label = 'Breiman Tree - Random sampling')
        bt_rn = axarr[1].plot(n_finals[:n_final_ind], BT_uc_MSE_temp[:n_final_ind], 
            color = 'green', linestyle = '--',label = 'Breiman Tree - Uncertainty sampling')
        axarr[1].legend(loc='best')

        f.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
        f.text(0.5, 0.01, 'Final number of labelled points', ha='center')

        plt.tight_layout()
        plt.savefig('graphs/sim_wine_uc_' + 
            str(len(data_seeds) * len(tree_seeds)) + '.pdf')


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
            MT_al.update_life_time((n_final**(1/(2+p))-1), set_seed=tree_seed)
            MT_rn = copy.deepcopy(MT_al)

            # print(MT_al._num_leaves)
            MT_al.input_data(X, range(n_start), y[:n_start])

            MT_al.make_full_leaf_list()
            MT_al.make_full_leaf_var_list()
            MT_al.al_set_default_var_global_var()
            # print(MT_al.al_default_var)

            MT_al.al_calculate_leaf_proportions()
            MT_al.al_calculate_leaf_number_new_labels(n_final)

            MT_uc = copy.deepcopy(MT_al)

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

            # MT_rn = Mondrian_Tree([[0,1]]*p)
            # MT_rn.update_life_time(n_final**(1/(2+p))-1, set_seed=tree_seed)
            # print(MT._num_leaves)
            MT_rn.input_data(X, range(n_final), y[:n_final])
            MT_rn.set_default_pred_global_mean()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_rn_preds = MT_rn.predict(X_test)
            MT_rn_preds = np.array(MT_rn_preds)
            MT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_rn_preds)**2)

            # print('Done MT_rn')

            # MT_uc

            new_labelled_points_uc = []
            MT_uc._al_proportions = [x / sum(MT_uc._full_leaf_var_list) for x in MT_uc._full_leaf_var_list]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_uc.al_calculate_leaf_number_new_labels(n_final)
            for i, node in enumerate(MT_uc._full_leaf_list):
                # print(i)
                
                num_new_points = MT_uc._al_leaf_number_new_labels[i]
                labels_to_add = node.pick_new_points(num_new_points,self_update = False, set_seed = tree_seed*i)
                
                new_labelled_points_uc.extend(labels_to_add)
                for ind in labels_to_add:
                    MT_uc.label_point(ind, y[ind])

            MT_uc.set_default_pred_global_mean()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                MT_uc_preds = MT_uc.predict(X_test)
            MT_uc_preds = np.array(MT_uc_preds)
            MT_uc_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - MT_uc_preds)**2)


            # BT_al

            BT_al = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = MT_al._num_leaves+1)
            BT_al.fit(X[list(range(n_start)) + new_labelled_points,:], y[list(range(n_start)) + new_labelled_points])
            BT_al_preds = BT_al.predict(X_test)
            BT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_al_preds)**2)
            # print('Done BT_al')

            BT_rn = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = MT_rn._num_leaves+1)
            BT_rn.fit(X[list(range(n_final)),:], y[list(range(n_final))])
            BT_rn_preds = BT_rn.predict(X_test)
            BT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_rn_preds)**2)

            # BT_uc
            BT_uc = DecisionTreeRegressor(random_state=tree_seed, max_leaf_nodes = MT_uc._num_leaves+1)
            BT_uc.fit(X[list(range(n_start)) + new_labelled_points_uc,:], y[list(range(n_start)) + new_labelled_points_uc])
            BT_uc_preds = BT_uc.predict(X_test)
            BT_uc_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_uc_preds)**2)
            # print('Done BT_rn')

MT_al_MSE = MT_al_MSE/(len(data_seeds) * len(tree_seeds))
MT_rn_MSE = MT_rn_MSE/(len(data_seeds) * len(tree_seeds))
MT_uc_MSE = MT_uc_MSE/(len(data_seeds) * len(tree_seeds))

BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))
BT_uc_MSE = BT_uc_MSE/(len(data_seeds) * len(tree_seeds))

np.savez('graphs/sim_wine_uc_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.npz', 
    MT_al_MSE=MT_al_MSE, MT_rn_MSE=MT_rn_MSE, 
    MT_uc_MSE=MT_uc_MSE, BT_uc_MSE=BT_uc_MSE,
    BT_al_MSE=BT_al_MSE, BT_rn_MSE=BT_rn_MSE)

f, axarr = plt.subplots(2, sharex=True)

mt_al = axarr[0].plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active sampling')
mt_rn = axarr[0].plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random sampling')
mt_uc = axarr[0].plot(n_finals, MT_uc_MSE, color = 'green', label = 'Mondrian Tree - Uncertainty sampling')
axarr[0].set_title('Wine experiment')
axarr[0].legend(loc='best')

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
plt.savefig('graphs/sim_wine_uc_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.pdf')