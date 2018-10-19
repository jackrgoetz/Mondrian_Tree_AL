from paper_var_forest_flux_para import *
from Mondrian_Forest import Mondrian_Forest
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

import numpy as np
import warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import copy
import math
import itertools
import time

data_seeds = [x * 11 for x in range(7)]
tree_seeds = [x * 13 for x in range(7)]

n_finals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

n_jobs = 4
verbose = 11

start = time.time()

results = Parallel(n_jobs=n_jobs,verbose=verbose)(delayed(cv_var_forest)(
    data_seed=data_seed,tree_seed=tree_seed) 
    for data_seed,tree_seed in itertools.product(data_seeds,tree_seeds))

finish = time.time() - start
print(finish)

MT_al_MSE = np.zeros([len(n_finals)])
MT_rn_MSE = np.zeros([len(n_finals)])
MT_uc_MSE = np.zeros([len(n_finals)])

BT_al_MSE = np.zeros([len(n_finals)])
BT_rn_MSE = np.zeros([len(n_finals)])
BT_uc_MSE = np.zeros([len(n_finals)])

for i in range(len(data_seeds) * len(tree_seeds)):
    MT_al_MSE += results[i][0]
    MT_rn_MSE += results[i][1]
    MT_uc_MSE += results[i][2]

    BT_al_MSE += results[i][3]
    BT_rn_MSE += results[i][4]
    BT_uc_MSE += results[i][5]

MT_al_MSE = MT_al_MSE/(len(data_seeds) * len(tree_seeds))
MT_rn_MSE = MT_rn_MSE/(len(data_seeds) * len(tree_seeds))
MT_uc_MSE = MT_uc_MSE/(len(data_seeds) * len(tree_seeds))

BT_al_MSE = BT_al_MSE/(len(data_seeds) * len(tree_seeds))
BT_rn_MSE = BT_rn_MSE/(len(data_seeds) * len(tree_seeds))
BT_uc_MSE = BT_uc_MSE/(len(data_seeds) * len(tree_seeds))

np.savez('graphs/sim_var_forest_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.npz', 
    MT_al_MSE=MT_al_MSE, MT_rn_MSE=MT_rn_MSE, 
    MT_uc_MSE=MT_uc_MSE, BT_uc_MSE=BT_uc_MSE,
    BT_al_MSE=BT_al_MSE, BT_rn_MSE=BT_rn_MSE)

f, axarr = plt.subplots(2, sharex=True)

mt_al = axarr[0].plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Forest - Active sampling')
mt_rn = axarr[0].plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Forest - Random sampling')
mt_uc = axarr[0].plot(n_finals, MT_uc_MSE, color = 'green', label = 'Mondrian Forest - Uncertainty sampling')
axarr[0].set_title('Varying complexity experiment')
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
plt.savefig('graphs/sim_var_forest_' + 
    str(len(data_seeds) * len(tree_seeds)) + '.pdf')