import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

n_finals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# n_finals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

mt_vals = dict(np.load('sim_ccpp_uc_2500.npz'))
# fr_vals = dict(np.load('sim_ccpp_forest_625.npz'))

# print(mt_vals)
# print(fr_vals)

f, axarr = plt.subplots(1)

mt_al = axarr.plot(n_finals, mt_vals['MT_al_MSE'], color = 'red', label='Mondrian Tree - Active sampling')
mt_rn = axarr.plot(n_finals, mt_vals['MT_rn_MSE'], color = 'blue', label = 'Mondrian Tree - Random sampling')
mt_uc = axarr.plot(n_finals, mt_vals['MT_uc_MSE'], color = 'green', label = 'Mondrian Tree - Uncertainty sampling')
axarr.set_title('CCPP experiment. m = 9568, d = 4', fontsize = 10)
# axarr[0].legend(loc='upper right')

# mf_al = axarr[1].plot(n_finals, mt_vals['BT_al_MSE'], color = 'red', linestyle = '--',
#  label='Mondrian Forest - Active sampling')
# mf_rn = axarr[1].plot(n_finals, mt_vals['BT_rn_MSE'], color = 'blue', linestyle = '--',
#  label = 'Mondrian Forest - Random sampling')
# mf_uc = axarr[1].plot(n_finals, mt_vals['BT_uc_MSE'], color = 'green', linestyle = '--',
#  label = 'Mondrian Forest - Uncertainty sampling')
# axarr[1].legend(loc='upper right')

# rf_al = axarr[2].plot(n_finals, fr_vals['BT_al_MSE'], color = 'red', linestyle = '-.',
#  label='Random Forest - Active sampling')
# rf_rn = axarr[2].plot(n_finals, fr_vals['BT_rn_MSE'], color = 'blue', linestyle = '-.',
#  label = 'Random Forest - Random sampling')
# rf_uc = axarr[2].plot(n_finals, fr_vals['BT_uc_MSE'], color = 'green', linestyle = '-.',
#  label = 'Random Forest - Uncertainty sampling')
# axarr[2].legend(loc='upper right')

f.text(0.0, 0.46, 'MSE', va='center', rotation='vertical', fontsize = 10)
f.text(0.5, 0.01, 'Final number of labelled points', ha='center', fontsize = 10)

# plt.tight_layout()
f.set_size_inches(4.5, 3)
plt.savefig('ccpp_final_mt.pdf')