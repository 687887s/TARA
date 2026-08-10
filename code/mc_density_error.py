import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

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
    
    # 4. 蒙地卡羅隨機擾動與標準高斯權重計算
    for _ in range(n_mc_realizations):
        # 僅擾動這 5 個鄰居的紅移
        perturbed_z = nearest_5_z_phot + np.random.normal(0, 1, size=len(nearest_5_z_phot)) * nearest_5_z_err
        
        # 使用標準高斯權重公式 [2]
        nearest_5_w = np.exp(-0.5 * ((perturbed_z - target_z) / nearest_5_z_err) ** 2)
        
        # 代入 5NN 表面環境密度公式 [3]
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