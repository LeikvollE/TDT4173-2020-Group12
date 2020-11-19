#!/usr/bin/env python3
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# from pprint import pp
import math
# from scipy.stats import gaussian_kde
# from matplotlib import cm
# from matplotlib.colors import Normalize 
from scipy.stats import linregress


accu = []
for i in range(100):
    with open('output/{:02d}'.format(i)) as f:
        accu.append(f.read().splitlines())

for i in range(100):
    with open('output.2020-11-13T05:36:49+01:00/{:02d}'.format(i)) as f:
        accu.append(f.read().splitlines())

# pp(accu)

# acculoss = {}

fig, ax = plt.subplots(8 * 2, 3)
fig.set_size_inches(6.4 * 3, 4.8 * 8 * 2)
fig.tight_layout(pad=4, h_pad=4, w_pad=4)

# fig.set(title='100 runs MNIST with euclidean distance')

loss_max = 0
loss_min = math.inf
acc_max = 0
acc_min = 1

for p in range(0, 800, 100):
    tmp1 = np.empty(200)
    tmp2 = np.empty(200)
    tmp3 = np.empty(200)
    other1 = np.empty(200)
    other2 = np.empty(200)
    other3 = np.empty(200)
    iter1 = np.empty(200)
    iter2 = np.empty(200)
    iter3 = np.empty(200)
    if p == 0:
        ax[int(p/100)][0].set(title='Random Vector')
        ax[int(p/100)][1].set(title='Forgy')
        ax[int(p/100)][2].set(title='Kmeans++')
    else:
        ax[int(p/100)][0].set(title='Random Vector, pca=%03d' % p)
        ax[int(p/100)][1].set(title='Forgy, pca=%03d' % p)
        ax[int(p/100)][2].set(title='Kmeans++, pca=%03d' % p)

    for i in range(200):
        if i < 100:
            if p == 0:
                basic = np.fromfile('output/%02d_basic_loss' % i)
                better = np.fromfile('output/%02d_better_loss' % i)
                kmeanspp = np.fromfile('output/%02d_kmeanspp_loss' % i)
            else:
                basic = np.fromfile(
                        'output/{:02d}_pca_{:d}_basic_loss'.format(i, p))
                better = np.fromfile(
                        'output/{:02d}_pca_{:d}_better_loss'.format(i, p))
                kmeanspp = np.fromfile(
                        'output/{:02d}_pca_{:d}_kmeanspp_loss'.format(i, p))
        else:
            if p == 0:
                basic = np.fromfile('output.2020-11-13T05:36:49+01:00/%02d_basic_loss' % (i - 100))
                better = np.fromfile('output.2020-11-13T05:36:49+01:00/%02d_better_loss' % (i - 100))
                kmeanspp = np.fromfile('output.2020-11-13T05:36:49+01:00/%02d_kmeanspp_loss' % (i - 100))
            else:
                basic = np.fromfile(
                        'output.2020-11-13T05:36:49+01:00/{:02d}_pca_{:d}_basic_loss'.format(i - 100, p))
                better = np.fromfile(
                        'output.2020-11-13T05:36:49+01:00/{:02d}_pca_{:d}_better_loss'.format(i - 100, p))
                kmeanspp = np.fromfile(
                        'output.2020-11-13T05:36:49+01:00/{:02d}_pca_{:d}_kmeanspp_loss'.format(i - 100, p))

        tmp1[i] = basic[-1]
        tmp2[i] = better[-1]
        tmp3[i] = kmeanspp[-1]

        iter1[i] = len(basic)
        iter2[i] = len(better)
        iter3[i] = len(kmeanspp)

        other1[i] = (accu[i].pop(0).split(' ')[-1])
        other2[i] = (accu[i].pop(0).split(' ')[-1])
        other3[i] = (accu[i].pop(0).split(' ')[-1])


    print('pca %d' % p)
    print('random:   mean {:003f}, median {:003f}, std {:003f}, var {:003f}, min {:003f}, max {:003f}'.format(np.mean(iter1), np.median(iter1), np.std(iter1), np.var(iter1), min(iter1), max(iter1)))
    print('forgy:    mean {:003f}, median {:003f}, std {:003f}, var {:003f}, min {:003f}, max {:003f}'.format(np.mean(iter2), np.median(iter2), np.std(iter2), np.var(iter2), min(iter2), max(iter2)))
    print('kmeanspp: mean {:003f}, median {:003f}, std {:003f}, var {:003f}, min {:003f}, max {:003f}'.format(np.mean(iter3), np.median(iter3), np.std(iter3), np.var(iter3), min(iter3), max(iter3)))

    # print(tmp1)
    # print(np.array(accu[i].pop(0).split(' ')[-1]))

    # ax[int(p/100)][0].scatter(np.array(accu[i].pop(0).split(' ')[-1]), tmp1)
    # ax[int(p/100)][1].scatter(np.array(accu[i].pop(0).split(' ')[-1]), tmp2)
    # ax[int(p/100)][2].scatter(np.array(accu[i].pop(0).split(' ')[-1]), tmp3)

    acc_max = max(acc_max, max(other1), max(other2), max(other3))
    acc_min = min(acc_min, min(other1), min(other2), min(other3))
    loss_max = max(loss_max, max(tmp1), max(tmp2), max(tmp3))
    loss_min = min(loss_min, min(tmp1), min(tmp2), min(tmp3))

    # xy = np.vstack([other1, tmp1])
    # z1 = gaussian_kde(xy)(xy)
    # xy = np.vstack([other2, tmp2])
    # z2 = gaussian_kde(xy)(xy)
    # xy = np.vstack([other3, tmp3])
    # z3 = gaussian_kde(xy)(xy)

    # ax[int(p/100)][0].scatter(other1, tmp1, c=z1, s=100, edgecolor='')
    # ax[int(p/100)][1].scatter(other2, tmp2, c=z2, s=100, edgecolor='')
    # ax[int(p/100)][2].scatter(other3, tmp3, c=z3, s=100, edgecolor='')

    # ax[int(p/100)][0].scatter(other1, tmp1)
    # ax[int(p/100)][1].scatter(other2, tmp2)
    # ax[int(p/100)][2].scatter(other3, tmp3)

    # pp(other1)
    # pp(tmp1)

    ax[int(p/100)*2][0].scatter(other1, tmp1)
    ax[int(p/100)*2][1].scatter(other2, tmp2)
    ax[int(p/100)*2][2].scatter(other3, tmp3)
    h = ax[int(p/100)*2+1][0].hist2d(other1, tmp1)
    fig.colorbar(h[3], ax=ax[int(p/100)*2+1][0])
    h = ax[int(p/100)*2+1][1].hist2d(other2, tmp2)
    fig.colorbar(h[3], ax=ax[int(p/100)*2+1][1])
    h = ax[int(p/100)*2+1][2].hist2d(other3, tmp3)
    fig.colorbar(h[3], ax=ax[int(p/100)*2+1][2])

    # lr1 = linregress(other1, tmp1)
    # lr2 = linregress(other2, tmp2)
    # lr3 = linregress(other3, tmp3)
    # ax[int(p/100)*2][0].plot(lr1)
    # ax[int(p/100)*2][1].plot(lr2)
    # ax[int(p/100)*2][2].plot(lr3)

    # lr1 = linregress(other1, tmp1)
    # print('pca %d' % p)
    # print('random	', end='')
    # print(lr1)
    # lr2 = linregress(other2, tmp2)
    # print('forgy	', end='')
    # print(lr2)
    # lr3 = linregress(other3, tmp3)
    # print('kmeans++	', end='')
    # print(lr3)
    # ax[int(p/100)*2][0].plot(other1, lr1[1] + lr1[0]*other1)
    # ax[int(p/100)*2][1].plot(other2, lr2[1] + lr2[0]*other2)
    # ax[int(p/100)*2][2].plot(other3, lr3[1] + lr3[0]*other3)

    # h = ax[int(p/100)][0].hist2d(other1, tmp1)
    # # fig.colorbar(h, ax=ax[int(p/100)][0])
    # h = ax[int(p/100)][1].hist2d(other2, tmp2)
    # # fig.colorbar(h, ax=ax[int(p/100)][1])
    # h = ax[int(p/100)][2].hist2d(other3, tmp3)
    # # fig.colorbar(h, ax=ax[int(p/100)][2])


    # density_scatter(other1, tmp1, ax=ax[int(p/100)][0], fig=fig)
    # density_scatter(other2, tmp2, ax=ax[int(p/100)][1], fig=fig)
    # density_scatter(other3, tmp3, ax=ax[int(p/100)][2], fig=fig)

    # ax[int(p/100)][0].scatter(*zip(*tmp1.sort(key=(lambda x: x[1]))))
    # ax[int(p/100)][1].scatter(*zip(*tmp2.sort(key=(lambda x: x[1]))))
    # ax[int(p/100)][2].scatter(*zip(*tmp3.sort(key=(lambda x: x[1]))))

for e in ax:
    for a in e:
        a.set(xlabel='Accuracy')
        a.set(ylabel='Loss')
        # a.label_outer()

# for a in range(16):
#     if not a % 2:
#         for e in ax[a]:
#             e.set_ylim(loss_min * 0.975, loss_max * 1.025)
#             e.set_xlim(acc_min * 0.975, acc_max * 1.025)

# for a in ax:
#     for e in a:
#         e.set_ylim(loss_min * 0.975, loss_max * 1.025)
#         e.set_xlim(acc_min * 0.975, acc_max * 1.025)

plt.show()
