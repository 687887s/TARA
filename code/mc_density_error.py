import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

import numpy as np

def calculate_single_target_fof_error(tg_ra, tg_dec, tg_z, tg_d, surround_ra, surround_dec, surround_z, surround_err, surround_d, Hz_surround, n_mc_realizations=100):
    """
    獨立的 FoF 蒙地卡羅誤差估算函數
    """
    all_ra = np.concatenate([[tg_ra], surround_ra])
    all_dec = np.concatenate([[tg_dec], surround_dec])
    
    ra_rad = np.radians(all_ra)
    dec_rad = np.radians(all_dec)
    
    # 預先計算夾角矩陣
    cos_theta_mat = (np.sin(dec_rad)[:, None] * np.sin(dec_rad)[None, :] +
                     np.cos(dec_rad)[:, None] * np.cos(dec_rad)[None, :] * 
                     np.cos(ra_rad[:, None] - ra_rad[None, :]))
    cos_theta_mat = np.clip(cos_theta_mat, -1.0, 1.0)
    theta_mat = np.arccos(cos_theta_mat)
    
    mc_richness = []
    
    for _ in range(n_mc_realizations):
        perturbed_z_surround = surround_z + np.random.normal(0, 1, size=len(surround_z)) * surround_err
        perturbed_z_surround = np.maximum(perturbed_z_surround, 0.0)
        
        # 泰勒展開快速距離估算
        dz_surround = perturbed_z_surround - surround_z
        perturbed_d_surround = surround_d + (299792.458 / Hz_surround) * dz_surround
        
        all_z = np.concatenate([[tg_z], perturbed_z_surround])
        all_d = np.concatenate([[tg_d], perturbed_d_surround])
        
        # BFS 滲流
        visited = np.zeros(len(all_ra), dtype=bool)
        queue = [0]
        visited[0] = True
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            unvisited_idx = np.where(~visited)[0]
            if len(unvisited_idx) == 0:
                break
                
            zi = all_z[curr]
            zj = all_z[unvisited_idx]
            z_mean = 0.5 * (zi + zj)
            delta_v = 299792.458 * np.abs(zi - zj) / (1.0 + z_mean)
            
            v_mask = delta_v <= 1000.0
            if not np.any(v_mask):
                continue
                
            candidates = unvisited_idx[v_mask]
            cos_theta = (np.sin(dec_rad[curr]) * np.sin(dec_rad[candidates]) +
                         np.cos(dec_rad[curr]) * np.cos(dec_rad[candidates]) * np.cos(ra_rad[curr] - ra_rad[candidates]))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            d_comoving_mean = 0.5 * (all_d[curr] + all_d[candidates])
            d_a = d_comoving_mean / (1.0 + 0.5 * (all_z[curr] + all_z[candidates]))
            d_perp = theta * d_a
            
            friends = candidates[d_perp <= 0.5]
            for f in friends:
                visited[f] = True
                queue.append(f)
                
        mc_richness.append(len(queue))
        
    error = np.std(mc_richness)
    return error

def calculate_single_target_5nn_error(
    target_z,
    bg_z_phot,
    bg_z_err,
    angular_separations, # 輸入單位：角分 (arcminutes)
    n_mc_realizations=100
):
    """
    計算單個目標星系的 5NN 環境密度誤差棒。
    
    此函數會自動將角分（arcmin）轉換為物理投影距離（Mpc），並在 5 個固定鄰居上進行 MC 擾動。
    """
    # 1. 使用 Planck18 計算目標紅移處的投影角直徑距離 D_A (Mpc/rad)
    da_target = cosmo.angular_diameter_distance(target_z).to_value(u.Mpc)
    
    # 2. 將角分 (arcmin) 轉為弧度 (rad)，再乘以 D_A 得到投影物理距離 (Mpc) [1]
    theta_rad = np.radians(angular_separations / 60.0)
    distances_mpc = da_target * theta_rad
    
    # 3. 鎖定物理距離最近的 5 個鄰居索引
    nearest_5_d = distances_mpc
    nearest_5_z_phot = bg_z_phot
    nearest_5_z_err = bg_z_err
    
    mc_log_sigma5 = []
    from scipy.stats import norm
    # 4. 蒙地卡羅隨機擾動與標準高斯權重計算
    for _ in range(n_mc_realizations):
        # 僅擾動這 5 個鄰居的紅移
        perturbed_z = nearest_5_z_phot + np.random.normal(0, 1, size=len(nearest_5_z_phot)) * nearest_5_z_err
        
        # 1. 定義靶星系紅移切片的物理邊界
        z_min = target_z - 1000/300000
        z_max = target_z + 1000/300000
        
        # 2. 直接使用標準正態分佈的 CDF 差值，計算在 [z_min, z_max] 區間內的物理概率
        nearest_5_w = norm.cdf(z_max, loc=perturbed_z, scale=nearest_5_z_err) - norm.cdf(z_min, loc=perturbed_z, scale=nearest_5_z_err)
        
        # 3. 代入 5NN 表面環境密度公式
        numerator = np.sum(nearest_5_w)
        denominator = np.pi * np.sum(nearest_5_w * (nearest_5_d ** 2))
        
        if denominator > 0:
            sigma_5 = numerator / denominator
            mc_log_sigma5.append(np.log10(sigma_5))
            
    # 5. 計算測光紅移不確定性傳播誤差 (err_photo_z)
    if len(mc_log_sigma5) > 1:
        err_photo_z = np.std(mc_log_sigma5)
    else:
        err_photo_z = np.nan
        
    # 6. 融合 5NN 固有泊松漲落誤差 (0.1942 dex)
    err_poisson = 0.4343 / np.sqrt(5.0)
    err_total = np.sqrt(err_photo_z**2 + err_poisson**2)
    
    return err_photo_z, err_total

def calculate_single_target_5nn_error_nonweighted(
    target_z,
    bg_z_phot,
    bg_z_err,
    angular_separations, # 輸入單位：角分 (arcminutes)
    n_mc_realizations=100
):
    """
    計算單個目標星系的 5NN 環境密度誤差棒。
    
    此函數會自動將角分（arcmin）轉換為物理投影距離（Mpc），並在 5 個固定鄰居上進行 MC 擾動。
    """
    # 1. 使用 Planck18 計算目標紅移處的投影角直徑距離 D_A (Mpc/rad)
    da_target = cosmo.angular_diameter_distance(target_z).to_value(u.Mpc)
    
    # 2. 將角分 (arcmin) 轉為弧度 (rad)，再乘以 D_A 得到投影物理距離 (Mpc) [1]
    theta_rad = np.radians(angular_separations / 60.0)
    distances_mpc = da_target * theta_rad
    
    # 3. 鎖定物理距離最近的 5 個鄰居索引
    nearest_5_d = distances_mpc
    nearest_5_z_phot = bg_z_phot
    nearest_5_z_err = bg_z_err
    
    mc_log_sigma5 = []
    from scipy.stats import norm
    # 4. 蒙地卡羅隨機擾動與標準高斯權重計算
    for _ in range(n_mc_realizations):
        # 僅擾動這 5 個鄰居的紅移
        perturbed_z = nearest_5_z_phot + np.random.normal(0, 1) * nearest_5_z_err
        
        # 1. 定義靶星系紅移切片的物理邊界
        z_min = target_z - 1000/300000
        z_max = target_z + 1000/300000
        
        # 2. 直接使用標準正態分佈的 CDF 差值，計算在 [z_min, z_max] 區間內的物理概率
        nearest_5_w = norm.cdf(z_max, loc=perturbed_z, scale=nearest_5_z_err) - norm.cdf(z_min, loc=perturbed_z, scale=nearest_5_z_err)
        
        # 3. 代入 5NN 表面環境密度公式
        numerator = 5
        denominator = np.pi * (nearest_5_d ** 2)
        
        if denominator > 0:
            sigma_5 = numerator / denominator
            mc_log_sigma5.append(np.log10(sigma_5))
            
    # 5. 計算測光紅移不確定性傳播誤差 (err_photo_z)
    if len(mc_log_sigma5) > 1:
        err_photo_z = np.std(mc_log_sigma5)
    else:
        err_photo_z = np.nan
        
    # 6. 融合 5NN 固有泊松漲落誤差 (0.1942 dex)
    err_poisson = 0.4343 / np.sqrt(5.0)
    err_total = np.sqrt(err_photo_z**2 + err_poisson**2)
    
    return err_photo_z, err_total

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 設置乾淨的學術繪圖風格
sns.set_theme(style='ticks', palette='colorblind')

def plot_density_with_errors(
    barred_log_sigma5, barred_err,
    unbarred_log_sigma5, unbarred_err
):
    sns.set_theme(style='ticks', palette='colorblind')

    """
    繪製有棒與無棒 NLS1 的環境密度分佈及誤差。
    
    參數:
    ----------
    barred_log_sigma5, unbarred_log_sigma5 : np.ndarray
        兩組樣本各自的 log10(Sigma_5) 密度值
    barred_err, unbarred_err : np.ndarray
        兩組樣本各自對應的總誤差 (err_total)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # =========================================================
    # 左圖：累積分佈函數 (CDF) 帶 $\pm 1\sigma$ 總誤差陰影帶
    # =========================================================
    
    # 1. 計算有棒樣本的 CDF
    b_sorted = np.sort(barred_log_sigma5)
    b_err_sorted = barred_err[np.argsort(barred_log_sigma5)]
    cdf_b = np.arange(1, len(b_sorted) + 1) / len(b_sorted)
    
    # 2. 計算無棒樣本的 CDF
    ub_sorted = np.sort(unbarred_log_sigma5)
    ub_err_sorted = unbarred_err[np.argsort(unbarred_log_sigma5)]
    cdf_ub = np.arange(1, len(ub_sorted) + 1) / len(ub_sorted)
    
    # 3. 繪製 CDF 實線
    ax1.plot(b_sorted, cdf_b, color='#0173b2', label='Barred NLS1', linewidth=2)
    ax1.plot(ub_sorted, cdf_ub, color='#de8f05', label='Unbarred NLS1', linewidth=2)
    
    # 4. 沿 X 軸（密度方向）繪製左右 $\pm 1\sigma$ 的誤差陰影帶
    ax1.fill_betweenx(cdf_b, b_sorted - b_err_sorted, b_sorted + b_err_sorted, 
                     color='#0173b2', alpha=0.15, label='Barred $\pm 1\sigma$ Band')
    ax1.fill_betweenx(cdf_ub, ub_sorted - ub_err_sorted, ub_sorted + ub_err_sorted, 
                     color='#de8f05', alpha=0.15, label='Unbarred $\pm 1\sigma$ Band')
    
    ax1.set_xlabel(r'$\log_{10}(\Sigma_5 \ [\mathrm{Mpc}^{-2}])$', fontsize=12)
    ax1.set_ylabel('Cumulative Fraction', fontsize=12)
    ax1.set_title('Cumulative Distribution with Uncertainty Band', fontsize=13, pad=12)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # =========================================================
    # 右圖：個體星系排序誤差棒圖 (Individual Error Bars)
    # =========================================================
    
    # 1. 繪製有棒星系（按密度從小到大排序）
    b_idx = np.arange(len(b_sorted))
    ax2.errorbar(b_idx, b_sorted, yerr=b_err_sorted, fmt='o', 
                 color='#0173b2', ecolor='#0173b2', elinewidth=1, capsize=2,
                 alpha=0.7, ms=4, label='Barred')
                 
    # 2. 繪製無棒星系（X座標平移以在視覺上分開兩組）
    ub_idx = np.arange(len(ub_sorted)) + len(b_sorted) + 5
    ax2.errorbar(ub_idx, ub_sorted, yerr=ub_err_sorted, fmt='o', 
                 color='#de8f05', ecolor='#de8f05', elinewidth=1, capsize=2,
                 alpha=0.7, ms=4, label='Unbarred')
                 
    ax2.set_xlabel('Galaxy Index (Sorted by Density)', fontsize=12)
    ax2.set_ylabel(r'$\log_{10}(\Sigma_5 \ [\mathrm{Mpc}^{-2}])$', fontsize=12)
    ax2.set_title('Individual Galaxy Densities with Error Bars', fontsize=13, pad=12)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pdf_with_errors(
    barred_log_sigma5, barred_err,
    unbarred_log_sigma5, unbarred_err,
    bins=10, range_limit=(-1.5, 2.5)
):
    sns.set_theme(style='ticks', palette='colorblind')
    """
    Plots the normalized PDF (histogram) of barred vs. unbarred NLS1 environments
    with propagated error bars on each bin.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Define bin edges
    bin_edges = np.linspace(range_limit[0], range_limit[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    # 2. Function to calculate binned PDF and its errors using MC realizations
    def get_binned_pdf_with_err(data, err_total, n_mc=200):
        n_stars = len(data)
        # We will perturb each data point within its err_total and re-bin it
        mc_histograms = np.zeros((n_mc, bins))
        
        for m in range(n_mc):
            # Perturb the calculated density of each galaxy by its total error
            perturbed_data = data + np.random.normal(0, 1, size=n_stars) * err_total
            counts, _ = np.histogram(perturbed_data, bins=bin_edges)
            # Normalize to make it a PDF (area under curve = 1)
            mc_histograms[m, :] = counts / (n_stars * bin_width)
            
        # The mean PDF and its standard deviation (error) in each bin
        mean_pdf = np.mean(mc_histograms, axis=0)
        err_pdf = np.std(mc_histograms, axis=0)
        return mean_pdf, err_pdf

    # 3. Calculate PDF and Errors for both samples
    pdf_b, err_b = get_binned_pdf_with_err(barred_log_sigma5, barred_err)
    pdf_ub, err_ub = get_binned_pdf_with_err(unbarred_log_sigma5, unbarred_err)
    
    # 4. Plot as step-histograms with error bars
    # Barred (Blue)
    ax.step(bin_edges[:-1], pdf_b, where='post', color='#0173b2', 
            label='Barred NLS1', linewidth=2)
    ax.errorbar(bin_centers, pdf_b, yerr=err_b, fmt='none', 
                ecolor='#0173b2', elinewidth=1.5, capsize=3)
    # Shading the area under the barred curve
    ax.fill_between(bin_edges[:-1], pdf_b, step="post", color='#0173b2', alpha=0.1)
    
    # Unbarred (Orange, slightly offset X to prevent overlapping error bars)
    offset = bin_width * 0.1
    ax.step(bin_edges[:-1] + offset, pdf_ub, where='post', color='#de8f05', 
            label='Unbarred NLS1', linewidth=2)
    ax.errorbar(bin_centers + offset, pdf_ub, yerr=err_ub, fmt='none', 
                ecolor='#de8f05', elinewidth=1.5, capsize=3)
    ax.fill_between(bin_edges[:-1] + offset, pdf_ub, step="post", color='#de8f05', alpha=0.1)
    
    # 5. Styling
    ax.set_xlabel(r'$\log_{10}(\Sigma_5 \ [\mathrm{Mpc}^{-2}])$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density (PDF)', fontsize=12, fontweight='bold')
    ax.set_title('Environmental Density Distribution (PDF)', fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(range_limit)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    sns.despine()
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_nls1_distribution(dens, ax=None, d=None, p=None, label_loc = 'upper right'):
    """
    繪製 100% 自動分箱、無邊界截斷且帶解析泊松誤差棒的直方圖。
    
    當傳入 ax（子圖）時，會自動取消標題設定，避免多子圖標題重疊。
    """
    # 1. 提取數據並轉為 float
    b_dens = dens[:, 0][dens[:, -1] == 'barred'].astype(float)
    ub_dens = dens[:, 0][dens[:, -1] == 'unbarred'].astype(float)

    # 2. 完全交給算法自動決定最優分箱邊界 (bins='auto')
    all_data = np.concatenate([b_dens, ub_dens])
    bins = np.histogram_bin_edges(all_data, bins='auto')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = np.diff(bins)  # 自動適應每個 bin 的實際寬度

    # 3. 判斷繪圖模式：是單圖還是子圖 (Subplot)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True
    else:
        is_standalone = False

    # 4. 繪製標準的 overlapping 直方圖並獲取歸一化高度 (density=True)
    b_heights, _, _ = ax.hist(b_dens, bins=bins, alpha=0.5, color='green', 
                               label='Barred', density=True)

    ub_heights, _, _ = ax.hist(ub_dens, bins=bins, alpha=0.5, color='red', 
                                label='Unbarred', density=True)

    # 5. 解析計算每個 bin 的歸一化泊松誤差
    counts_b, _ = np.histogram(b_dens, bins=bins)
    counts_ub, _ = np.histogram(ub_dens, bins=bins)

    b_bin_errs = np.sqrt(counts_b) / (len(b_dens) * bin_width)
    ub_bin_errs = np.sqrt(counts_ub) / (len(ub_dens) * bin_width)

    # 6. 繪製誤差棒（左右微調 10% bin 寬度，防止重疊）
    ax.errorbar(bin_centers - bin_width * 0.1, b_heights, yerr=b_bin_errs, 
                fmt='none', ecolor='green', elinewidth=1.5, capsize=3)

    ax.errorbar(bin_centers + bin_width * 0.1, ub_heights, yerr=ub_bin_errs, 
                fmt='none', ecolor='red', elinewidth=1.5, capsize=3)

    text_placeholder = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible = False)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([text_placeholder])
    handles.extend([text_placeholder])
    labels.extend([f'K-S Test p-value: {p:.4f}'])
    labels.extend([f'K-S Test d-value: {d:.4f}'])

    ax.legend(handles= handles, labels = labels, fontsize=8, loc=label_loc)
    ax.grid(True, linestyle=':', alpha=0.5)

    # 8. 僅在獨立繪圖模式下設置標題並執行 show()
    if is_standalone:
        ax.set_title('Environmental Density Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()