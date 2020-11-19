#!/usr/bin/env python3
import numpy as np
# from cycler import cycler
# import matplotlib
import matplotlib.pyplot as plt
import math

fig, ax = plt.subplots(1, 3)

# fig.set_size_inches(6.4 * 3, 4.8 * 8)
fig.set_size_inches(6.4 * 3, 4.8)
fig.tight_layout(pad=4, h_pad=4, w_pad=4)

# fig.set_size_inches(8.27, 11.7)
# fig.tight_layout(pad=1, h_pad=1, w_pad=1)

print(ax)

loss_max = 0
loss_min = math.inf

ax[0].set(title='basic')
ax[1].set(title='better')
ax[2].set(title='kmeanspp')
for i in range(100):  # for best of 99, for 32 change 100 to 33
    basic = np.fromfile('stars_output_manhattan/%02d_basic_loss' % i)
    better = np.fromfile('stars_output_manhattan/%02d_better_loss' % i)
    kmeanspp = np.fromfile('stars_output_manhattan/%02d_kmeanspp_loss' % i)

    loss_max = max(loss_max, max(basic), max(better), max(kmeanspp))
    loss_min = min(loss_min, min(basic), min(better), min(kmeanspp))

    ax[0].plot(basic)
    ax[1].plot(better)
    ax[2].plot(kmeanspp)

# for a in ax:
#     for e in a:
#         e.set_ylim(loss_min * 0.975, loss_max * 1.025)

# plt.legend()
plt.show()
# fig.savefig('bb_loss_euclid.pgf')

# fig, ax = plt.subplots()
# ax.plot(loss)
# fig.show()

