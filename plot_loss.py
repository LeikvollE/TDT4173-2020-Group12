#!/usr/bin/env python3
import numpy as np
# from cycler import cycler
# import matplotlib
import matplotlib.pyplot as plt
import math

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

# fig, ax = plt.subplots()
# ax.set_prop_cycle(cc)

# NUM_COLORS = 32

# fig, ax = plt.subplots()
# # cm = plt.get_cmap('gist_rainbow')
# # ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# cm = plt.get_cmap('tab20c')
# # ax.set_prop_cycle('color', [cm(i) for i in range(20)])
# cc = (cycler(color=[cm(i) for i in range(20)]) * cycler(linestyle=['-', '--']))
# ax.set_prop_cycle(cc)

fig, ax = plt.subplots(8, 3)

fig.set_size_inches(6.4 * 3, 4.8 * 8)
fig.tight_layout(pad=4, h_pad=4, w_pad=4)

# fig.set_size_inches(8.27, 11.7)
# fig.tight_layout(pad=1, h_pad=1, w_pad=1)

print(ax)

loss_max = 0
loss_min = math.inf

for p in range(0, 800, 100):
    if p == 0:
        ax[int(p/100)][0].set(title='basic')
        ax[int(p/100)][1].set(title='better')
        ax[int(p/100)][2].set(title='kmeanspp')
    else:
        ax[int(p/100)][0].set(title='basic, pca=%03d' % p)
        ax[int(p/100)][1].set(title='better, pca=%03d' % p)
        ax[int(p/100)][2].set(title='kmeanspp, pca=%03d' % p)
    for i in range(100):
        if p == 0:
            basic = np.fromfile('output/%02d_basic_loss' % i)
            better = np.fromfile('output/%02d_better_loss' % i)
            kmeanspp = np.fromfile('output/%02d_kmeanspp_loss' % i)
        else:
            basic = np.fromfile('output/{:02d}_pca_{:d}_basic_loss'.format(i, p))
            better = np.fromfile('output/{:02d}_pca_{:d}_better_loss'.format(i, p))
            kmeanspp = np.fromfile('output/{:02d}_pca_{:d}_kmeanspp_loss'.format(i, p))

        loss_max = max(loss_max, max(basic), max(better), max(kmeanspp))
        loss_min = min(loss_min, min(basic), min(better), min(kmeanspp))

        ax[int(p/100)][0].plot(basic)
        ax[int(p/100)][1].plot(better)
        ax[int(p/100)][2].plot(kmeanspp)

for a in ax:
    for e in a:
        e.set_ylim(loss_min * 0.975, loss_max * 1.025)

# plt.legend()
plt.show()
# fig.savefig('bb_loss_euclid.pgf')

# fig, ax = plt.subplots()
# ax.plot(loss)
# fig.show()
