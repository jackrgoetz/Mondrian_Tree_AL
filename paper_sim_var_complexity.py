from data_sets.toy_data_var_complexity import toy_data_var_complexity
from Mondrian_Tree import Mondrian_Tree
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import warnings
import matplotlib.pyplot as plt

n_points = 20000
n_test_points = 500
n_finals = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# n_finals = [2000]
p = 5

data_seeds = [x * 11 for x in range(5)]
tree_seeds = [x * 13 for x in range(5)]

std = 1
low_freq = 0.1
high_freq = 0.05

low_mag = 1
high_mag = 5

high_area = [[0.5,1]]*p

MT_al_MSE = np.zeros([len(n_finals)])
MT_rn_MSE = np.zeros([len(n_finals)])
MT_oracle_MSE = np.zeros([len(n_finals)])
BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])

for n_final_ind, n_final in enumerate(n_finals):

    n_start = int(n_final/2)

    for data_seed in data_seeds:

        X, y = toy_data_var_complexity(n=n_points,p=p,high_area=high_area,std=std,
            low_freq=low_freq,high_freq=high_freq, low_mag=low_mag, high_mag=high_mag, 
            set_seed=data_seed)

        X = np.array(X)
        y = np.array(y)

        # plt.scatter(X[:,0], X[:,1], c=y)
        # plt.show()
        # sys.exit()

        np.random.seed(data_seed)

        cv_ind = np.random.permutation(range(X.shape[0]))

        train_ind_al = cv_ind[:n_start]
        train_ind_rn = cv_ind[:n_final]

        X = X[cv_ind,:]
        y = y[cv_ind]

        X_test, y_test = toy_data_var_complexity(n=n_points,p=p,high_area=high_area,std=std,
            low_freq=low_freq,high_freq=high_freq, low_mag=low_mag, high_mag=high_mag, 
            set_seed=data_seed+1)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

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

plt.title('Varying complexity simulation')
plt.xlabel('Final number of labelled points')
plt.ylabel('MSE')
# plt.show()

# plt.clf()

plt.plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Breiman Tree - Active labelling')
plt.plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Breiman Tree - Random labelling')
plt.legend(loc="best")
plt.savefig('graphs/sim_var_complexity_big.pdf')
plt.show()

plt.clf()
plt.plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active labelling')
plt.plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random labelling')
plt.savefig('graphs/sim_var_complexity_big_MT.pdf')
plt.show()

plt.clf()
plt.plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Breiman Tree - Active labelling')
plt.plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Breiman Tree - Random labelling')
plt.savefig('graphs/sim_var_complexity_big_BT.pdf')
plt.show()