import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime
import time

# --- データ取得関数 ---
def get_jma_data(prec_no, block_no, year, month, day):
    url = f"https://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no={prec_no}&block_no={block_no}&year={year}&month={month}&day={day}&view="
    try:
        r = requests.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.find_all('tr', class_='mtx')
        data = []
        for row in rows[2:]:
            cols = row.find_all('td')
            data.append([col.text for col in cols])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# --- 設定 ---
stations = {'tokyo': [44, 47662], 'kofu': [49, 47638]}
# 取得したいターゲット： [年, 月, 日数]
target_months = [[2023, 12, 31], [2025, 1, 31]]
all_data_list = []

print("🚀 不足データの追加取得を開始します...")

# --- ループ処理 ---
for year, month, days in target_months:
    for day in range(1, days + 1):
        daily_combined = pd.DataFrame()
        
        for name, ids in stations.items():
            df_day = get_jma_data(ids[0], ids[1], year, month, day)
            
            if not df_day.empty:
                # 1時間ごとのデータを数値化して集計
                df_day.columns = ['時', '現地気圧', '海面気圧', '降水量', '気温', '露点温度', '蒸気圧', '湿度', '平均風速', '風向', '最大瞬間風速', '最大瞬間風速風向', '日照時間', '全天日射量', '降雪', '積雪', '天気']
                for col in ['気温', '湿度', '海面気圧', '降水量', '日照時間']:
                    df_day[col] = pd.to_numeric(df_day[col], errors='coerce')
                
                summary = {
                    f'{name}_temp_mean': df_day['気温'].mean(),
                    f'{name}_temp_max': df_day['気温'].max(),
                    f'{name}_temp_min': df_day['気温'].min(),
                    f'{name}_hum': df_day['湿度'].mean(),
                    f'{name}_press': df_day['海面気圧'].mean(),
                    f'{name}_precip': df_day['降水量'].sum(),
                    f'{name}_sun': df_day['日照時間'].sum()
                }
                
                temp_df = pd.DataFrame([summary])
                if daily_combined.empty:
                    daily_combined = temp_df
                else:
                    daily_combined = pd.concat([daily_combined, temp_df], axis=1)
        
        if not daily_combined.empty:
            daily_combined['date'] = datetime.date(year, month, day)
            all_data_list.append(daily_combined)
        
        print(f" ✅ {year}年{month}月{day}日 完了")
        time.sleep(0.5)

# --- CSVに保存 ---
df_extra = pd.concat(all_data_list, ignore_index=True)
df_extra.to_csv('weather_data_extra.csv', index=False)
print("\n✨ 'weather_data_extra.csv' の作成が完了しました！")