import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kstest
from mc_density_error import plot_nls1_distribution

llr_dens = np.load('/data/TARA/code/llr_dens.npy', allow_pickle=True)
xgb_dens = np.load('/data/TARA/code/xgb_dens.npy', allow_pickle=True)
STRM_dens = np.load('/data/TARA/code/STRM_dens.npy', allow_pickle=True)

# low, high = np.percentile(STRM_dens[:, 0].astype(np.float64), [0.5, 99.5])
# STRM_dens = STRM_dens[(STRM_dens[:, 0].astype(np.float64) >= low) & (STRM_dens[:, 0].astype(np.float64) <= high)]
# low, high = np.percentile(xgb_dens[:, 0].astype(np.float64), [0.5, 99.5])
# xgb_dens = xgb_dens[(xgb_dens[:, 0].astype(np.float64) >= low) & (xgb_dens[:, 0].astype(np.float64) <= high)]
# low, high = np.percentile(llr_dens[:, 0].astype(np.float64), [0.5, 99.5])
# llr_dens = llr_dens[(llr_dens[:, 0].astype(np.float64) >= low) & (llr_dens[:, 0].astype(np.float64) <= high)]

fig, ax = plt.subplots(2,2, figsize=(8, 5))
d, p = kstest(xgb_dens[:, 0][xgb_dens[:, -1] == 'barred'].astype(np.float64), xgb_dens[:, 0][xgb_dens[:, -1] == 'unbarred'].astype(np.float64))
plot_nls1_distribution(xgb_dens, ax[0,0], d, p, label_loc = 'upper right')
ax[0,0].set_title('XGB Density Distribution')

d, p = kstest(llr_dens[:, 0][llr_dens[:, -1] == 'barred'].astype(np.float64), llr_dens[:, 0][llr_dens[:, -1] == 'unbarred'].astype(np.float64))
plot_nls1_distribution(llr_dens, ax[0,1], d, p, label_loc = 'upper right')
ax[0,1].set_title('LLR Density Distribution')

d, p = kstest(STRM_dens[:, 0][STRM_dens[:, -1] == 'barred'].astype(np.float64), STRM_dens[:, 0][STRM_dens[:, -1] == 'unbarred'].astype(np.float64))
plot_nls1_distribution(STRM_dens, ax[1,0], d, p, label_loc = 'upper right')
ax[1,0].set_title('STRM Density Distribution')

ax[1,1].hist(xgb_dens[:, 0].astype(np.float64), bins=20, label='XGB', color='green', alpha=0.3, density=True)
ax[1,1].hist(llr_dens[:, 0].astype(np.float64), bins=20, label='LLR', color='blue', alpha=0.3, density=True)
ax[1,1].hist(STRM_dens[:, 0].astype(np.float64), bins=20, label='STRM', color='red', alpha=0.3, density=True)
ax[1,1].set_title('Histogram of Density Values')
ax[1,1].legend(loc='upper right', fontsize=8)
from scipy.stats import pearsonr

llr_dens = np.load('/data/TARA/code/llr_dens.npy', allow_pickle=True)
xgb_dens = np.load('/data/TARA/code/xgb_dens.npy', allow_pickle=True)
STRM_dens = np.load('/data/TARA/code/STRM_dens.npy', allow_pickle=True)

r0, p0 = pearsonr(llr_dens[:, 0].astype(np.float64), xgb_dens[:, 0].astype(np.float64))
r1, p1 = pearsonr(llr_dens[:, 0].astype(np.float64), STRM_dens[:, 0].astype(np.float64))
print(r0, p0)
print(r1, p1)

fig.supxlabel('Density (5th nn density)', fontsize = 12)
fig.supylabel('Frequency', fontsize = 12)

plt.tight_layout()
plt.savefig(f'/data/TARA/fig/density_histogram.png', dpi=300)
plt.show()