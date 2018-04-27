# Create toy data with constan mean but heteroskedasticity

import random

def toy_data_var(n,p,high_area,constant=0,low_std=1,high_std=3, 
    set_seed=None, marginal = 'uniform'):

    if set_seed is not None:
        random.seed(set_seed)

    points = []
    labels = []
    for i in range(n):
        point = []
        is_high_var = []
        for j in range(p):
            if marginal == 'normal':
                val = random.gauss(0.5, 0.5/3)
            else:
                val = random.random()

            point.append(val)
            if val > high_area[j][0] and val < high_area[j][1]:
                is_high_var.append(True)
            else:
                is_high_var.append(False)

        if all(is_high_var):
            labels.append(constant + random.gauss(0, high_std))
        else:
            labels.append(constant + random.gauss(0, low_std))
        points.append(point)


    return points, labels


