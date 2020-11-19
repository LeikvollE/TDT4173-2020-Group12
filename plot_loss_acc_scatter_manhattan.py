#!/usr/bin/env python3
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import math
# from pprint import pp


accu = []
for i in range(100):
    with open('output_manhattan/{:02d}'.format(i)) as f:
        accu.append(f.read().splitlines())

fig, ax = plt.subplots(8 * 2, 3)
fig.set_size_inches(6.4 * 3, 4.8 * 8 * 2)
fig.tight_layout(pad=4, h_pad=4, w_pad=4)

# fig.set(title='100 runs MNIST with manhattan distance')

loss_max = 0
loss_min = math.inf
acc_max = 0
acc_min = 1

for p in range(0, 800, 100):
    tmp1 = np.empty(100)
    tmp2 = np.empty(100)
    tmp3 = np.empty(100)
    other1 = np.empty(100)
    other2 = np.empty(100)
    other3 = np.empty(100)
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
            basic = np.fromfile('output_manhattan/%02d_basic_loss' % i)
            better = np.fromfile('output_manhattan/%02d_better_loss' % i)
            kmeanspp = np.fromfile('output_manhattan/%02d_kmeanspp_loss' % i)
        else:
            basic = np.fromfile(
                    'output_manhattan/{:02d}_pca_{:d}_basic_loss'.format(i, p))
            better = np.fromfile(
                    'output_manhattan/{:02d}_pca_{:d}_better_loss'.format(i, p))
            kmeanspp = np.fromfile(
                    'output_manhattan/{:02d}_pca_{:d}_kmeanspp_loss'.format(i, p))

        tmp1[i] = basic[-1]
        tmp2[i] = better[-1]
        tmp3[i] = kmeanspp[-1]

        other1[i] = (accu[i].pop(0).split(' ')[-1])
        other2[i] = (accu[i].pop(0).split(' ')[-1])
        other3[i] = (accu[i].pop(0).split(' ')[-1])

    acc_max = max(acc_max, max(other1), max(other2), max(other3))
    acc_min = min(acc_min, min(other1), min(other2), min(other3))
    loss_max = max(loss_max, max(tmp1), max(tmp2), max(tmp3))
    loss_min = min(loss_min, min(tmp1), min(tmp2), min(tmp3))

    # ax[int(p/100)][0].scatter(other1, tmp1)
    # ax[int(p/100)][1].scatter(other2, tmp2)
    # ax[int(p/100)][2].scatter(other3, tmp3)

    # ax[int(p/100)][0].hist2d(other1, tmp1)
    # ax[int(p/100)][1].hist2d(other2, tmp2)
    # ax[int(p/100)][2].hist2d(other3, tmp3)

    ax[int(p/100)*2][0].scatter(other1, tmp1)
    ax[int(p/100)*2][1].scatter(other2, tmp2)
    ax[int(p/100)*2][2].scatter(other3, tmp3)
    h = ax[int(p/100)*2+1][0].hist2d(other1, tmp1)
    fig.colorbar(h[3], ax=ax[int(p/100)*2+1][0])
    h = ax[int(p/100)*2+1][1].hist2d(other2, tmp2)
    fig.colorbar(h[3], ax=ax[int(p/100)*2+1][1])
    h = ax[int(p/100)*2+1][2].hist2d(other3, tmp3)
    fig.colorbar(h[3], ax=ax[int(p/100)*2+1][2])

for e in ax:
    for a in e:
        a.set(xlabel='Accuracy')
        a.set(ylabel='Loss')
        # a.label_outer()

for a in range(16):
    if not a % 2:
        for e in ax[a]:
            e.set_ylim(loss_min * 0.975, loss_max * 1.025)
            e.set_xlim(acc_min * 0.975, acc_max * 1.025)

# for a in ax:
#     for e in a:
#         e.set_ylim(loss_min * 0.975, loss_max * 1.025)
#         e.set_xlim(acc_min * 0.975, acc_max * 1.025)

plt.show()
