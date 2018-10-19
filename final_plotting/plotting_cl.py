import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

# n_finals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
n_finals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

mt_vals = dict(np.load('sim_cl_uc_2500.npz'))
fr_vals = dict(np.load('sim_cl_forest_900.npz'))

# print(mt_vals)
# print(fr_vals)

f, axarr = plt.subplots(3, sharex=True)

mt_al = axarr[0].plot(n_finals, mt_vals['MT_al_MSE'], color = 'red', label='Mondrian Tree - Active sampling')
mt_rn = axarr[0].plot(n_finals, mt_vals['MT_rn_MSE'], color = 'blue', label = 'Mondrian Tree - Random sampling')
mt_uc = axarr[0].plot(n_finals, mt_vals['MT_uc_MSE'], color = 'green', label = 'Mondrian Tree - Uncertainty sampling')
axarr[0].set_title('Cl experiment. m = 1518, d = 12', fontsize = 14)
# axarr[0].legend(loc='upper right')

mf_al = axarr[1].plot(n_finals, fr_vals['MT_al_MSE'], color = 'red', linestyle = '--',
 label='Mondrian Forest - Active sampling')
mf_rn = axarr[1].plot(n_finals, fr_vals['MT_rn_MSE'], color = 'blue', linestyle = '--',
 label = 'Mondrian Forest - Random sampling')
mf_uc = axarr[1].plot(n_finals, fr_vals['MT_uc_MSE'], color = 'green', linestyle = '--',
 label = 'Mondrian Forest - Uncertainty sampling')
# axarr[1].legend(loc='upper right')

rf_al = axarr[2].plot(n_finals, fr_vals['BT_al_MSE'], color = 'red', linestyle = '-.',
 label='Random Forest - Active sampling')
rf_rn = axarr[2].plot(n_finals, fr_vals['BT_rn_MSE'], color = 'blue', linestyle = '-.',
 label = 'Random Forest - Random sampling')
rf_uc = axarr[2].plot(n_finals, fr_vals['BT_uc_MSE'], color = 'green', linestyle = '-.',
 label = 'Random Forest - Uncertainty sampling')
# axarr[2].legend(loc='upper right')

f.text(0.0, 0.46, 'MSE', va='center', rotation='vertical', fontsize = 14)
f.text(0.5, 0.01, 'Final number of labelled points', ha='center', fontsize = 14)

# plt.tight_layout()
f.set_size_inches(4.5, 4.5)
plt.savefig('cl_final.pdf')