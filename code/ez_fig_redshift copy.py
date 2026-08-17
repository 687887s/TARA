import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import pearsonr

strm_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
llr_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
xgb_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')

low, high = np.percentile(strm_rd['z_phot0'], [0.5, 99.5])
strm_rd = strm_rd[(strm_rd['z_phot0'] >= low) & (strm_rd['z_phot0'] <= high)]
low, high = np.percentile(llr_rd['z_phot'], [0.5, 99.5])
llr_rd = llr_rd[(llr_rd['z_phot'] >= low) & (llr_rd['z_phot'] <= high)]
low, high = np.percentile(xgb_rd['z_phot_xgb_origin'], [0.5, 99.5])
xgb_rd = xgb_rd[(xgb_rd['z_phot_xgb_origin'] >= low) & (xgb_rd['z_phot_xgb_origin'] <= high)]

r0, p0 = pearsonr(llr_rd['z_phot'], llr_rd['z_phot'])
r1, p1 = pearsonr(llr_rd['z_phot'], xgb_rd['z_phot'])
r2, p2 = pearsonr(llr_rd['z_phot'], strm_rd['z_phot0'])

plt.scatter(llr_rd['z_phot'], llr_rd['z_phot'], label='LLR redshift distribution', alpha = 0.3)
plt.scatter(llr_rd['z_phot'], xgb_rd['z_phot_xgb_origin'], label='XGB redshift distribution', alpha = 0.3)
plt.scatter(llr_rd['z_phot'], strm_rd['z_phot0'], label='STRM redshift distribution', alpha = 0.3)
# plt.hist(sdss_rd['zsp'], bins=20, label='SDSS redshift distribution', color='red', alpha=0.5, density=True)
ax = plt.gca()
text_placeholder = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible = False)
handles, labels = ax.get_legend_handles_labels()
handles.extend([text_placeholder])
handles.extend([text_placeholder])
handles.extend([text_placeholder])
labels.extend([f'llr Test p-value: {p0:.4f}, r-value: {r0:.4f}'])
labels.extend([f'xgb p-value: {p1:.4f}, r-value: {r1:.4f}'])
labels.extend([f'strm Test p-value: {p2:.4f},r-value: {r2:.4f}'])
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshift Values')
plt.legend(handles = handles, labels = labels, fontsize = 8, loc = 'upper right')
plt.tight_layout()
plt.savefig('/data/TARA/fig/redshift_distribution_histogram.png', dpi=300)
plt.show()