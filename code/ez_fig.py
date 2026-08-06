import numpy as np
import matplotlib.pyplot as plt

llr_dens = np.load('/data/TARA/code/llr_dens.npy', allow_pickle=True)
xgb_dens = np.load('/data/TARA/code/xgb_dens.npy', allow_pickle=True)
STRM_dens = np.load('/data/TARA/code/STRM_dens.npy', allow_pickle=True)

fig, ax = plt.subplots(2,2, figsize=(8, 5))
ax[0,0].hist(xgb_dens[0][xgb_dens[1] == 'barred'], bins=20, label='barred', color='green', alpha=0.5, density=True)
ax[0,0].hist(xgb_dens[0][xgb_dens[1] == 'unbarred'], bins=20, label='unbarred', color='red', alpha=0.5, density=True)
ax[0,0].set_title('XGB Density Distribution')
ax[0,0].set_xlabel('Density (5th nearest neighbor)')
ax[0,0].set_ylabel('Frequency')
ax[0,0].legend(loc='upper right')

ax[0,1].hist(llr_dens[0][llr_dens[1] == 'barred'], bins=20, label='barred', color='green', alpha=0.5, density=True)
ax[0,1].hist(llr_dens[0][llr_dens[1] == 'unbarred'], bins=20, label='unbarred', color='red', alpha=0.5, density=True)
ax[0,1].set_title('LLR Density Distribution')
ax[0,1].set_xlabel('Density (5th nearest neighbor)')
ax[0,1].set_ylabel('Frequency')
ax[0,1].legend(loc='upper right')

ax[1,0].hist(STRM_dens[0][STRM_dens[1] == 'barred'], bins=20, label='barred', color='green', alpha=0.5, density=True)
ax[1,0].hist(STRM_dens[0][STRM_dens[1] == 'unbarred'], bins=20, label='unbarred', color='red', alpha=0.5, density=True)
ax[1,0].set_title('STRM Density Distribution')
ax[1,0].set_xlabel('Density (5th nearest neighbor)')
ax[1,0].set_ylabel('Frequency')
ax[1,0].legend(loc='upper right')

ax[1,1].hist(xgb_dens[0][0], bins=20, label='XGB density distribution', color='green', alpha=0.3, density=True)
ax[1,1].hist(llr_dens[0][0], bins=20, label='LLR density distribution', color='blue', alpha=0.3, density=True)
ax[1,1].hist(STRM_dens[0], bins=20, label='STRM density distribution', color='red', alpha=0.3, density=True)
ax[1,1].set_xlabel('Density (5th nearest neighbor)')
ax[1,1].set_ylabel('Frequency')
ax[1,1].set_title('Histogram of Density Values')
ax[1,1].legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'/data/TARA/fig/density_histogram.png', dpi=300)
plt.show()