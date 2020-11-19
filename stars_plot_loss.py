#!/usr/bin/env python3
import numpy as np
# from cycler import cycler
# import matplotlib
import matplotlib.pyplot as plt
import math

fig, ax = plt.subplots(5, 3)

# fig.set_size_inches(6.4 * 3, 4.8 * 8)
fig.set_size_inches(6.4 * 3, 4.8 * 5)
fig.tight_layout(pad=4, h_pad=4, w_pad=4)

# fig.set_size_inches(8.27, 11.7)
# fig.tight_layout(pad=1, h_pad=1, w_pad=1)

print(ax)

loss_max = 0
loss_min = math.inf

for p in range(5):
    if p == 0:
        ax[p][0].set(title='Random Vector')
        ax[p][1].set(title='Forgy')
        ax[p][2].set(title='Kmeans++')
    else:
        ax[p][0].set(title='Random Vector, PCA=%d' % p)
        ax[p][1].set(title='Forgy, PCA=%d' % p)
        ax[p][2].set(title='Kmeans++, PCA=%d' % p)
    for i in range(1000):
        if p == 0:
            basic = np.fromfile('stars_output/%03d_basic_loss' % i)
            better = np.fromfile('stars_output/%03d_better_loss' % i)
            kmeanspp = np.fromfile('stars_output/%03d_kmeanspp_loss' % i)
        else:
            basic = np.fromfile('stars_output/{:03d}_pca_{:d}_basic_loss'.format(i, p))
            better = np.fromfile('stars_output/{:03d}_pca_{:d}_better_loss'.format(i, p))
            kmeanspp = np.fromfile('stars_output/{:03d}_pca_{:d}_kmeanspp_loss'.format(i, p))

        loss_max = max(loss_max, max(basic), max(better), max(kmeanspp))
        loss_min = min(loss_min, min(basic), min(better), min(kmeanspp))

        ax[p][0].plot(basic)
        ax[p][1].plot(better)
        ax[p][2].plot(kmeanspp)

for a in ax:
    for e in a:
        e.set_ylim(loss_min * 0.975, loss_max * 1.025)

# plt.legend()
plt.show()
# fig.savefig('bb_loss_euclid.pgf')

# fig, ax = plt.subplots()
# ax.plot(loss)
# fig.show()

