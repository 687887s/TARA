import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from astropy.cosmology import Planck18
import requests
from io import StringIO
from astropy.io import fits
from astropy.visualization import simple_norm
import os
from glob import glob
from SciServer import CasJobs

df = pd.read_csv('../all_sample_20260702.csv')

z_value = df['paliya_Z'].values
df = df.rename(columns = {"paliya_RA":"RA", "paliya_DEC":"DEC"})

r = 5*u.Mpc
size_ang = Planck18.arcsec_per_kpc_proper(z_value)*  r.to(u.kpc)
size_ang = size_ang.to(u.degree)
size_ang = size_ang.value
size = size_ang*3600*4

import time
import numpy as np
import mastcasjobs
import pandas as pd
from astropy.table import Table
from glob import glob

# 請務必確認帳號密碼
jobs_api = mastcasjobs.MastCasJobs(username="yee8787", password="xafUQJdC@Ga6JwR", context="HLSP_PS1_STRM")

if __name__ == "__main__":
    valid = []
    for i, (r, d) in enumerate(zip(df['RA'], df['DEC'])):
        if glob('../data/catalog/ps1/redshift_STRM/t{:08.4f}{:+07.4f}.csv'.format(r, d)) == []:
            valid.append(i)
    valid = np.array(valid)
    np.save('./valid.npy', valid)
    
    if len(valid) > 0:
        print(f"準備提交 {len(valid)} 筆星體的查詢任務...")
        
        job_records = []
        
        # ==========================================
        # 階段一：逐一送出所有查詢請求 (Submit)
        # ==========================================
        for idx in valid:
            ra = df['RA'].iloc[idx]
            dec = df['DEC'].iloc[idx]
            radius_deg = size_ang[idx]
            
            # 建立每個查詢專屬的資料表名稱
            results_table = f"ps1_res_{int(time.time())}_{idx}"
            
            sql = f"""
            SELECT
                p.objID,
                p.raMean,
                p.decMean,
                p.z_phot0,
                p.z_photErr,
                r.distance AS distance_arcmin
            INTO MyDB.{results_table}
            FROM 
                dbo.fGetNearbyObjEq(CAST({ra} AS FLOAT), CAST({dec} AS FLOAT), CAST({radius_deg * 60} AS FLOAT)) r
            JOIN
                catalogRecordRowStore p ON p.objID = r.objID
            WHERE p.extrapolation_Photoz = 0
            AND p.extrapolation_Class = 0
            AND p.class = 'GALAXY'
            AND p.z_phot >= 0
            ORDER BY 
                p.raMean ASC
            """

            
            # 提交任務
            job_id = jobs_api.submit(sql, task_name=f"PS1_Task_{idx}")
            job_records.append({
                "job_id": job_id,
                "table": results_table,
                "ra": ra,
                "dec": dec
            })
            print(f"已提交任務 RA:{ra:.4f}, DEC:{dec:.4f} (Job ID: {job_id})")
            
            # 暫停 1 秒，避免連續送太多請求被封鎖
            time.sleep(1)
            
        print("\n所有任務已成功送往排程！開始進行狀態檢查與下載...\n")
        
        # ==========================================
        # 階段二：逐一下載結果並刪除伺服器暫存檔
        # ==========================================
        for record in job_records:
            job_id = record["job_id"]
            results_table = record["table"]
            ra = record["ra"]
            dec = record["dec"]
            
            # 1. 輪詢直到該任務完成
            while True:
                try:
                    status = jobs_api.status(job_id)[0]
                    if status == 5:
                        break
                    elif status in [3, 4, 6]:
                        print(f"任務失敗或已取消 (Job ID: {job_id})")
                        break
                except Exception:
                    # 避免 GetJobStatus 偶發的 500 報錯
                    pass
                time.sleep(3)
                
            # 2. 如果成功完成，進行下載與過濾
            try:
                if status == 5:
                    tab_result = jobs_api.get_table(results_table)
                    if len(tab_result) > 0:
                        df_result = tab_result.to_pandas()
                        
                        if len(df_result) > 0:
                            fname = "../data/catalog/ps1/redshift_STRM/t{:08.4f}{:+07.4f}.csv".format(ra, dec)
                            df_result = df_result.drop(columns=['target_ra', 'target_dec'], errors='ignore')
                            final_tab = Table.from_pandas(df_result)
                            final_tab.write(fname, format="csv", overwrite=True)
                            print(f"✅ 成功下載並儲存: RA:{ra:.4f}, DEC:{dec:.4f} (共 {len(df_result)} 筆)")
                        else:
                            print(f"過濾後無符合資料: RA:{ra:.4f}, DEC:{dec:.4f}")
                    else:
                        print(f"無鄰近資料: RA:{ra:.4f}, DEC:{dec:.4f}")
                        
            except Exception as e:
                print(f"下載過程發生錯誤 RA:{ra:.4f}, DEC:{dec:.4f} - 錯誤: {e}")
            finally:
                # 3. 確保無論如何都會把伺服器上的資料表刪除 (清理戰場)
                try:
                    jobs_api.drop_table_if_exists(results_table)
                except Exception:
                    pass
                    
        print("\n所有流程已完成並清理完畢！")
    else:
        print("All targets have already been downloaded.")