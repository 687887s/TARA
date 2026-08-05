import numpy as np
import matplotlib.pyplot as plt

llr_dens = np.load('/data/TARA/code/llr_dens.npy')
xgb_dens = np.load('/data/TARA/code/xgb_dens.npy')
STRM_dens = np.load('/data/TARA/code/STRM_dens.npy')

plt.figure(figsize=(8, 5))
plt.hist(xgb_dens, bins=20, label='XGB density distribution', color='green', alpha=0.5, density=True)
plt.hist(llr_dens, bins=20, label='LLR density distribution', color='blue', alpha=0.5, density=True)
plt.hist(STRM_dens, bins=20, label='STRM density distribution', color='orange', alpha=0.5, density=True)
# plt.axvline(-3.0, color='green', linestyle='dashed', linewidth=1, label='XGB median')
# plt.text(-2.9, 2, 'STRM density is much lower than others', verticalalignment='bottom', color='green')
plt.xlabel('Density (5th nearest neighbor)')
plt.ylabel('Frequency')
plt.title('Histogram of Density Values')
plt.legend()
plt.tight_layout()
plt.savefig(f'/data/TARA/fig/density_histogram.png', dpi=300)
plt.show()