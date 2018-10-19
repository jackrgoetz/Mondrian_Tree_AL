from data_sets.toy_data_var import toy_data_var
from Breiman_Tree import Breiman_Tree
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy

n_points = 10000
n_test_points = 5000
n_finals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
# n_finals = [100, 200]
p = 10
marginal = 'normal'
batch_size = 50

# n_finals = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# p = 5

data_seeds = [x * 11 for x in range(10)]
tree_seeds = [x * 13 for x in range(10)]

constant = 0
low_std = 1
high_std = 5

high_area = [[0.5,1]]*p

BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])
BT_uc_MSE = np.zeros([len(n_finals)])

for n_final_ind, n_final in enumerate(n_finals):

    n_start = int(n_final/2)

    for data_seed in data_seeds:

        X, y = toy_data_var(n=n_points,p=p,high_area=high_area,constant=constant,
            low_std=low_std,high_std=high_std, set_seed=data_seed, marginal=marginal)

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

        X_test, y_test = toy_data_var(n=n_test_points,p=p,high_area=high_area,constant=constant,
            low_std=low_std,high_std=high_std, set_seed=data_seed+1,marginal=marginal)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        for tree_seed in tree_seeds:

            print(n_final, data_seed, tree_seed)

            # BT_al

            BT_al = Breiman_Tree(seed = tree_seed, min_samples_leaf=10)
            BT_al.input_data(X, range(n_start), y[:n_start])
            for _ in range(int((n_final - n_start)/batch_size)):
                # print(_)
                BT_al.fit_tree()
                BT_al.al_calculate_leaf_proportions()
                new_points = BT_al.pick_new_points(num_samples = batch_size)
                for new_point in new_points:
                    BT_al.label_point(new_point, y[new_point])
            
            BT_al_preds = BT_al.predict(X_test)
            BT_al_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_al_preds)**2)
            # print('Done BT_al')

            # BT_rn

            BT_rn = DecisionTreeRegressor(random_state=tree_seed, min_samples_leaf=10)
            BT_rn.fit(X[list(range(n_final)),:], y[list(range(n_final))])
            BT_rn_preds = BT_rn.predict(X_test)
            BT_rn_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_rn_preds)**2)
            # print('Done BT_rn')

            # BT_uc
            BT_uc = Breiman_Tree(seed = tree_seed, min_samples_leaf=10)
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

            BT_uc_preds = BT_uc.predict(X_test)
            BT_uc_MSE[n_final_ind] += sum(1/X_test.shape[0]*(y_test - BT_uc_preds)**2)

BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))
BT_uc_MSE = BT_uc_MSE/(len(data_seeds) * len(tree_seeds))

np.savez('graphs/sim_heteroskedastic_BT_' + str(p) + '_' + 
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

bt_al = axarr[0].plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--', 
    label = 'Breiman Tree - Active sampling')
bt_rn = axarr[0].plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
    label = 'Breiman Tree - Random sampling')
bt_rn = axarr[0].plot(n_finals, BT_uc_MSE, color = 'green', linestyle = '--',
    label = 'Breiman Tree - Uncertainty sampling')
axarr[1].legend(loc='best')

f.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
f.text(0.5, 0.01, 'Final number of labelled points', ha='center')

plt.tight_layout()
plt.savefig('graphs/sim_heteroskedastic_BT_' + str(p) + '_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.pdf')