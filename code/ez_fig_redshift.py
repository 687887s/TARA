import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

strm_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
llr_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
xgb_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/t000.4369+11.1667.csv')
sdss_rd = pd.read_csv('/data/TARA/data/catalog/ps1/matched/sdss/t000.4369+11.1667.csv')

low, high = np.percentile(strm_rd['z_phot0'], [0.5, 99.5])
strm_rd = strm_rd[(strm_rd['z_phot0'] >= low) & (strm_rd['z_phot0'] <= high)]
low, high = np.percentile(llr_rd['z_phot'], [0.5, 99.5])
llr_rd = llr_rd[(llr_rd['z_phot'] >= low) & (llr_rd['z_phot'] <= high)]
low, high = np.percentile(xgb_rd['z_phot_xgb_origin'], [0.5, 99.5])
xgb_rd = xgb_rd[(xgb_rd['z_phot_xgb_origin'] >= low) & (xgb_rd['z_phot_xgb_origin'] <= high)]
low, high = np.percentile(sdss_rd['z_phot'], [0.5, 99.5])
sdss_rd = sdss_rd[(sdss_rd['z_phot'] >= low) & (sdss_rd['z_phot'] <= high)]

plt.hist(strm_rd['z_phot0'], bins=20, label='STRM redshift distribution', color='orange', alpha=0.5, density=True)
plt.hist(llr_rd['z_phot'], bins=20, label='LLR redshift distribution', color='blue', alpha=0.5, density=True)
plt.hist(xgb_rd['z_phot_xgb_origin'], bins=20, label='XGB redshift distribution', color='green', alpha=0.5, density=True)
plt.hist(sdss_rd['z_phot'], bins=20, label='SDSS redshift distribution', color='red', alpha=0.5, density=True)
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshift Values')
plt.legend()
plt.tight_layout()
plt.savefig('/data/TARA/fig/redshift_distribution_histogram.png', dpi=300)
plt.show()