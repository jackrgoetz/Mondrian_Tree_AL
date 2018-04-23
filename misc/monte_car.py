from __future__ import division
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# t dist

# para = 5
# samp_var = 1* para/(para-2)

# n = 10
# reps = 10000

# counter = 0
# ests = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = np.random.standard_t(para, n)
#     if np.var(draw, ddof=1) < samp_var:
#         counter += 1
#         ests.append(np.mean(draw)**2)
#         print_counter = True

# print(np.mean(ests))
# print(para/(para-2) * 1/n) 

# beta dist

# alpha = 10
# beta = 10

# samp_var = 1* alpha*beta / ((alpha + beta)**2 * (alpha + beta + 1))

# n = 10
# reps = 10000

# counter = 0
# ests = []
# means = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = np.random.beta(alpha, beta, n)
#     # if np.var(draw, ddof=1) < samp_var:
#     if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
#         counter += 1
#         means.append(np.mean(draw))
#         ests.append((np.mean(draw) - (alpha/(alpha+beta)) )**2)
#         # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
#         #     print(draw)
#         print_counter = True

# print(np.mean(ests))
# print( alpha*beta / ((alpha + beta)**2 * (alpha + beta + 1) )* 1/n) 

# plt.hist(means, bins = 50)
# plt.show()

# double exp

# alpha = 10

# samp_var = 1.5* 2*alpha**2

# n = 10
# reps = 10000

# counter = 0
# ests = []
# means = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = np.random.laplace(0, alpha, n)
#     if np.var(draw, ddof=1) < samp_var:
#     # if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
#         counter += 1
#         means.append(np.mean(draw))
#         ests.append((np.mean(draw))**2)
#         # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
#         #     print(draw)
#         print_counter = True

# print(np.mean(ests))
# print( 2*alpha**2 * 1/n) 

# plt.hist(means, bins = 50)
# plt.show()

# double gamma

# alpha = 0.1
# beta = 1
# samp_var = 0.5* alpha*beta**2

# n = 10
# reps = 10000

# counter = 0
# ests = []
# means = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = np.random.gamma(alpha, beta, n) * np.random.choice([-1,1], n)
#     if np.var(draw, ddof=1) < samp_var:
#     # if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
#         counter += 1
#         means.append(np.mean(draw))
#         ests.append((np.mean(draw))**2)
#         # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
#         #     print(draw)
#         print_counter = True

# print(np.mean(ests))
# # print( alpha*beta**2 * 1/n) 

# plt.hist(means, bins = 50)
# plt.show()

# uniform

# alpha = -1
# beta = 1
# samp_var = 1* 4/12

# n = 10
# reps = 10000

# counter = 0
# ests = []
# means = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = np.random.uniform(alpha, beta, n) #* np.random.choice([-1,1], n)
#     if np.var(draw, ddof=1) > samp_var:
#     # if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
#         counter += 1
#         means.append(np.mean(draw))
#         ests.append((np.mean(draw))**2)
#         # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
#         #     print(draw)
#         print_counter = True

# print(np.mean(ests))
# print( 4/12 * 1/n) 

# plt.hist(means, bins = 50)
# plt.show()

# arcsign

# alpha = -np.pi
# beta = np.pi
# samp_var = 1* 4*1/8

# n = 10
# reps = 10000

# counter = 0
# ests = []
# means = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = np.sin(np.random.uniform(alpha, beta, n)) #* np.random.choice([-1,1], n)
#     if np.var(draw, ddof=1) > samp_var:
#     # if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
#         counter += 1
#         means.append(np.mean(draw))
#         ests.append((np.mean(draw))**2)
#         # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
#         #     print(draw)
#         print_counter = True

# print(np.mean(ests))
# print( 4/8 * 1/n) 

# plt.hist(means, bins = 50)
# plt.show()

# hyperbolic secant

# samp_var = 1* 1

# n = 10
# reps = 10000

# counter = 0
# ests = []
# means = []
# print_counter = True
# while counter < reps:

#     if counter % 100 == 0 and print_counter:
#         print(counter)
#         print_counter = False

#     draw = st.hypsecant.rvs(size=n) #* np.random.choice([-1,1], n)
#     # print(np.var(draw, ddof=1))
#     if np.var(draw, ddof=1) > samp_var:
#     # if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
#         counter += 1
#         means.append(np.mean(draw))
#         ests.append((np.mean(draw))**2)
#         # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
#         #     print(draw)
#         print_counter = True

# print(np.mean(ests))

# plt.hist(means, bins = 50)
# plt.show()

# Mixtures of normals

# compare (0.95, -2, 2, 1, 1) to (0.59, -2,2,1,1). Both have same s,k but one has ratio < 1 and other >1
# (0.9, 1,-1,1,1) have an example where k < 0 but ratio <1

p1 = 0.9
p2 = 1-p1

mu1 = 1
mu2 = -1

std1 = 1
std2 = 1

mu = p1*mu1 + p2*mu2

samp_var = 1* (p1*std1**2 + p2*std2**2 + p1*mu1**2 + p2*mu2**2 - (p1*mu1 + p2*mu2)**2)

n = 10
reps = 1000000

counter = 0
ests = []
means = []
print_counter = True
while counter < reps:

    if counter % 1000 == 0 and print_counter:
        print(counter)
        print_counter = False

    draw1 = np.random.normal(mu1,std1,n) #* np.random.choice([-1,1], n)
    draw2 = np.random.normal(mu2,std2,n)
    which_dist = np.random.choice([1,0], size = n, p = [p1,p2])
    draw = which_dist * draw1 + (1-which_dist) * draw2
    # print(np.round(draw1,1))
    # print(np.round(draw2,1))
    # print(np.round(draw,1))
    # print(np.var(draw, ddof=1))
    if np.var(draw, ddof=1) < samp_var:
    # if abs(np.var(draw, ddof=1) - samp_var) < 0.1 * samp_var:
        counter += 1
        means.append(np.mean(draw))
        ests.append((np.mean(draw)-mu)**2)
        # if np.mean(draw) < 0.3 or np.mean(draw) > 0.7:
        #     print(draw)
        print_counter = True
s = ((p1*(mu1-mu)*(3*std1**2 + (mu1-mu)**2) + p2*(mu2-mu)*(3*std2**2 + (mu2-mu)**2))/
    (p1*(mu1**2 + std1**2) + p2*(mu2**2 + std2**2))**(3/2))
k = ((p1 * (mu1**4 + 6*mu1**2*std1**2 + 3*std1**4) + p2 * (mu2**4 + 6*mu2**2*std2**2 + 3*std2**4))/
    (p1*(mu1**2 + std1**2) + p2*(mu2**2 + std2**2))**2) - 3
print('Skewness is {}'.format(s))
print('Excess Kurtosis is {}'.format(k))
print(np.mean(ests))
print( (p1*std1**2 + p2*std2**2 + p1*mu1**2 + p2*mu2**2 - (p1*mu1 + p2*mu2)**2) * 1/n )
print(np.mean(ests) / ((p1*std1**2 + p2*std2**2 + p1*mu1**2 + p2*mu2**2 - (p1*mu1 + p2*mu2)**2) * 1/n))

# plt.hist(means, bins = 50)
# plt.show()