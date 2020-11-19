#!/usr/bin/env python3
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# from pprint import pp
import math
# from scipy.stats import gaussian_kde
# from matplotlib import cm
# from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


accu = []

for i in range(1000):
    with open('stars_output_manhattan/{:03d}'.format(i)) as f:
        accu.append(f.read().splitlines())

# pp(accu)

# acculoss = {}

fig, ax = plt.subplots(10, 3)
fig.set_size_inches(6.4 * 3, 4.8 * 10)  # * 8)
fig.tight_layout(pad=4, h_pad=4, w_pad=4)

loss_max = 0
loss_min = math.inf
acc_max = 0
acc_min = 1

for p in range(5):
    tmp1 = np.empty(1000)
    tmp2 = np.empty(1000)
    tmp3 = np.empty(1000)
    other1 = np.empty(1000)
    other2 = np.empty(1000)
    other3 = np.empty(1000)
    if p == 0:
        ax[p][0].set(title='Random Vector')
        ax[p][1].set(title='Forgy')
        ax[p][2].set(title='Kmeans++')
    else:
        ax[p*2][0].set(title='Random Vector, pca=%03d' % p)
        ax[p*2][1].set(title='Forgy, pca=%03d' % p)
        ax[p*2][2].set(title='Kmeans++, pca=%03d' % p)

    for i in range(1000):
        if p == 0:
            basic = np.fromfile('stars_output_manhattan/%03d_basic_loss' % i)
            better = np.fromfile('stars_output_manhattan/%03d_better_loss' % i)
            kmeanspp = np.fromfile('stars_output_manhattan/%03d_kmeanspp_loss' % i)
        else:
            basic = np.fromfile('stars_output_manhattan/{:03d}_pca_{:d}_basic_loss'.format(i, p))
            better = np.fromfile('stars_output_manhattan/{:03d}_pca_{:d}_better_loss'.format(i, p))
            kmeanspp = np.fromfile('stars_output_manhattan/{:03d}_pca_{:d}_kmeanspp_loss'.format(i, p))

        tmp1[i-1] = basic[-1]
        tmp2[i-1] = better[-1]
        tmp3[i-1] = kmeanspp[-1]

        other1[i-1] = (accu[i-1].pop(0).split(' ')[-1])
        other2[i-1] = (accu[i-1].pop(0).split(' ')[-1])
        other3[i-1] = (accu[i-1].pop(0).split(' ')[-1])

    acc_max = max(acc_max, max(other1), max(other2), max(other3))
    acc_min = min(acc_min, min(other1), min(other2), min(other3))
    loss_max = max(loss_max, max(tmp1), max(tmp2), max(tmp3))
    loss_min = min(loss_min, min(tmp1), min(tmp2), min(tmp3))

    ax[p*2][0].scatter(other1, tmp1)
    ax[p*2][1].scatter(other2, tmp2)
    ax[p*2][2].scatter(other3, tmp3)
    h = ax[p*2+1][0].hist2d(other1, tmp1)
    fig.colorbar(h[3], ax=ax[p*2+1][0])
    h = ax[p*2+1][1].hist2d(other2, tmp2)
    fig.colorbar(h[3], ax=ax[p*2+1][1])
    h = ax[p*2+1][2].hist2d(other3, tmp3)
    fig.colorbar(h[3], ax=ax[p*2+1][2])

for e in ax:
    for a in e:
        a.set(xlabel='Accuracy')
        a.set(ylabel='Loss')
        # a.label_outer()

# fig.colorbar()

# for a in ax:
#     for e in a:
#         e.set_ylim(loss_min * 0.975, loss_max * 1.035)
#         e.set_xlim(acc_min * 0.975, acc_max * 1.035)

plt.show()

