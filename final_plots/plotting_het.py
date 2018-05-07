import numpy as np
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

n_finals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# n_finals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

mt_vals = dict(np.load('sim_heteroskedastic_uc_10_2500.npz'))
fr_vals = dict(np.load('sim_het_forest_484.npz'))

# print(mt_vals)
# print(fr_vals)

f, axarr = plt.subplots(3, sharex=True)

mt_al = axarr[0].plot(n_finals, mt_vals['MT_al_MSE'], color = 'red', label='MT - Active')
mt_rn = axarr[0].plot(n_finals, mt_vals['MT_rn_MSE'], color = 'blue', label = 'MT - Random')
mt_uc = axarr[0].plot(n_finals, mt_vals['MT_uc_MSE'], color = 'green', label = 'MT - Uncertainty')
axarr[0].set_title('Heteroskedastic simulation', fontsize = 14)
axarr[0].legend(loc='upper right', fontsize = 8)

mf_al = axarr[1].plot(n_finals, fr_vals['MT_al_MSE'], color = 'red', linestyle = '--',
 label='MF - Active')
mf_rn = axarr[1].plot(n_finals, fr_vals['MT_rn_MSE'], color = 'blue', linestyle = '--',
 label = 'MF - Random')
mf_uc = axarr[1].plot(n_finals, fr_vals['MT_uc_MSE'], color = 'green', linestyle = '--',
 label = 'MF - Uncertainty')
axarr[1].legend(loc='upper right', fontsize = 8)

rf_al = axarr[2].plot(n_finals, fr_vals['BT_al_MSE'], color = 'red', linestyle = '-.',
 label='RF - Active')
rf_rn = axarr[2].plot(n_finals, fr_vals['BT_rn_MSE'], color = 'blue', linestyle = '-.',
 label = 'RF - Random')
rf_uc = axarr[2].plot(n_finals, fr_vals['BT_uc_MSE'], color = 'green', linestyle = '-.',
 label = 'RF - Uncertainty')
axarr[2].legend(loc='upper right', fontsize = 8)

f.text(0.00, 0.46, 'MSE', va='center', rotation='vertical', fontsize = 14)
f.text(0.5, 0.01, 'Final number of labelled points', ha='center', fontsize = 14)

# plt.tight_layout()
f.set_size_inches(4.5, 4.5)
# plt.show()
plt.savefig('het_final.pdf')