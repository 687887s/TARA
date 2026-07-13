import os
import glob
import pandas as pd

# 定義數據目錄與目標合併檔案路徑
# 預設數據位於 ../data/redshift/，合併後存放在 ../data/merged_redshifts.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "catalog"))
OUTPUT_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "merged_catalog.csv"))

def merge_csv_files():
    # 使用 glob 獲取所有符合 pattern 的 csv 檔案
    search_pattern = os.path.join(DATA_DIR, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    print(f"在 {DATA_DIR} 中找到 {len(csv_files)} 個 CSV 檔案。")
    if not csv_files:
        print("未找到任何 CSV 檔案，請確認路徑是否有誤。")
        return

    # 讀取並合併所有的 csv 檔案
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"已讀取: {os.path.basename(file)}")
        except Exception as e:
            print(f"讀取檔案 {file} 時發生錯誤: {e}")

    if df_list:
        # 合併 DataFrames
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # 確保目標資料夾存在
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        # 儲存合併後的 CSV 檔案
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n合併成功！合併後的檔案已儲存至: {OUTPUT_FILE}")
        print(f"總共列數: {len(merged_df)}")
    else:
        print("沒有成功讀取任何資料，未產生合併檔案。")

if __name__ == "__main__":
    merge_csv_files()
