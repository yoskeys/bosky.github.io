import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime

# --- 1. データ取得関数 ---
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
        print(f"Error at {month}/{day}: {e}")
        return pd.DataFrame()

# --- 2. 設定 ---
stations = {
    'tokyo': [44, 47662],
    'kofu': [49, 47638]
}
year = 2024
all_data_list = []

print(f"🚀 {year}年のデータ収集を開始します。")
print("※1年分の取得には約15分〜20分かかります。ゆっくりお待ちください。")

# --- 3. 2重ループで1年分回す ---
for month in range(1, 13):
    # 各月の日数を判定（うるう年も自動考慮）
    if month in [4, 6, 9, 11]:
        days_in_month = 30
    elif month == 2:
        days_in_month = 29 # 2024年はうるう年
    else:
        days_in_month = 31

    for day in range(1, days_in_month + 1):
        daily_combined = pd.DataFrame()
        
        for name, ids in stations.items():
            df_day = get_jma_data(ids[0], ids[1], year, month, day)
            
            if not df_day.empty:
                # 必要な列を数値化して集計
                df_day.columns = ['時', '現地気圧', '海面気圧', '降水量', '気温', '露点温度', '蒸気圧', '湿度', '平均風速', '風向', '最大瞬間風速', '最大瞬間風速風向', '日照時間', '全天日射量', '降雪', '積雪', '天気']
                for col in ['気温', '湿度', '海面気圧', '降水量', '日照時間']:
                    df_day[col] = pd.to_numeric(df_day[col], errors='coerce')
                
                # 集計（平均、最高、最低、合計など）
                summary = {
                    f'{name}_temp_mean': df_day['気温'].mean(),
                    f'{name}_temp_max': df_day['気温'].max(),
                    f'{name}_temp_min': df_day['気温'].min(),
                    f'{name}_hum': df_day['湿度'].mean(),
                    f'{name}_press': df_day['海面気圧'].mean(),
                    f'{name}_precip': df_day['降水量'].sum(), # 降水量合計
                    f'{name}_sun': df_day['日照時間'].sum()     # 日照時間合計
                }
                
                # 地点データを一時保存
                temp_df = pd.DataFrame([summary])
                if daily_combined.empty:
                    daily_combined = temp_df
                else:
                    daily_combined = pd.concat([daily_combined, temp_df], axis=1)
        
        if not daily_combined.empty:
            daily_combined['date'] = datetime.date(year, month, day)
            all_data_list.append(daily_combined)
        
        print(f" ✅ {month}月{day}日 完了")
        time.sleep(0.5) # サーバーへの優しさ

# --- 4. CSVに保存 ---
df_final = pd.concat(all_data_list, ignore_index=True)
df_final.to_csv('weather_data_2024_full.csv', index=False)

print("\n✨ 1年分のデータを取得し、'weather_data_2024_full.csv' に保存しました！")