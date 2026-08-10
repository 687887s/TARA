import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import kstest

strm_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
llr_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
xgb_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')

low, high = np.percentile(strm_rd['z_phot0'], [0.5, 99.5])
strm_rd = strm_rd[(strm_rd['z_phot0'] >= low) & (strm_rd['z_phot0'] <= high)]
low, high = np.percentile(llr_rd['z_phot'], [0.5, 99.5])
llr_rd = llr_rd[(llr_rd['z_phot'] >= low) & (llr_rd['z_phot'] <= high)]
low, high = np.percentile(xgb_rd['z_phot_xgb_origin'], [0.5, 99.5])
xgb_rd = xgb_rd[(xgb_rd['z_phot_xgb_origin'] >= low) & (xgb_rd['z_phot_xgb_origin'] <= high)]

d0, p0 = kstest(llr_rd['z_phot'], llr_rd['z_phot'])
d1, p1 = kstest(llr_rd['z_phot'], xgb_rd['z_phot'])
d2, p2 = kstest(llr_rd['z_phot'], strm_rd['z_phot0'])

plt.hist(llr_rd['z_phot'], bins=20, label='LLR redshift distribution', fill = False, alpha = 1, density=True, edgecolor = 'blue', linewidth = 1, histtype='stepfilled')
plt.hist(xgb_rd['z_phot_xgb_origin'], bins=20, label='XGB redshift distribution', fill = False, alpha = 1, density=True, edgecolor = 'green', linewidth = 1, histtype='stepfilled')
plt.hist(strm_rd['z_phot0'], bins=20, label='STRM redshift distribution', fill = False, alpha = 1, density=True, edgecolor = 'orange', linewidth = 1, histtype='stepfilled')
# plt.hist(sdss_rd['zsp'], bins=20, label='SDSS redshift distribution', color='red', alpha=0.5, density=True)
ax = plt.gca()
text_placeholder = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible = False)
handles, labels = ax.get_legend_handles_labels()
handles.extend([text_placeholder])
handles.extend([text_placeholder])
handles.extend([text_placeholder])
labels.extend([f'llr Test p-value: {p0:.4f}, d-value: {d0:.4f}'])
labels.extend([f'xgb p-value: {p1:.4f}, d-value: {d1:.4f}'])
labels.extend([f'strm Test p-value: {p2:.4f}, d-value: {d2:.4f}'])
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshift Values')
plt.legend(handles = handles, labels = labels, fontsize = 8, loc = 'upper right')
plt.tight_layout()
plt.savefig('/data/TARA/fig/redshift_distribution_histogram.png', dpi=300)
plt.show()