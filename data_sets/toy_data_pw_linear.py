# Create toy data with constant variance but changing OLS slope (same slope value for all dim)

import random

def toy_data_pw_linear(n,p,high_area,high_constant=3, low_constant=1,std=1, set_seed=None):

    if set_seed is not None:
        random.seed(set_seed)

    points = []
    labels = []
    for i in range(n):
        point = []
        is_high_slope = []
        for j in range(p):
            val = random.random()
            point.append(val)
            if val > high_area[j][0] and val < high_area[j][1]:
                is_high_slope.append(True)
            else:
                is_high_slope.append(False)

        if all(is_high_slope):
            labels.append(high_constant*sum(point) + random.gauss(0, std))
        else:
            labels.append(low_constant*sum(point) + random.gauss(0, std))
        points.append(point)


    return points, labels


