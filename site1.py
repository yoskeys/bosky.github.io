import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import pickle

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
    except:
        return pd.DataFrame()

# --- 2. 設定とデータ収集 ---
stations = {
    'tokyo': [44, 47662],
    'kofu': [49, 47638]
}
all_stations_daily = []

for name, ids in stations.items():
    station_data = []
    for day in range(1, 32):
        print(f" 📡 {name}: {day}日目を取得中...")
        df_day = get_jma_data(ids[0], ids[1], 2023, 1, day)
        if not df_day.empty:
            station_data.append(df_day)
        time.sleep(1) # サーバー負荷軽減
    
    if not station_data:
        continue

    # 取得データの整形
    df_st = pd.concat(station_data, ignore_index=True)
    df_st.columns = ['時', '現地気圧', '海面気圧', '降水量', '気温', '露点温度', '蒸気圧', '湿度', '平均風速', '風向', '最大瞬間風速', '最大瞬間風速風向', '日照時間', '全天日射量', '降雪', '積雪', '天気']
    
    for col in ['気温', '湿度', '海面気圧']:
        df_st[col] = pd.to_numeric(df_st[col], errors='coerce')
    
    # ★ここが重要：平均・最高・最低をすべて抽出
    df_daily = df_st.groupby(df_st.index // 24).agg({
        '気温': ['mean', 'max', 'min'],
        '湿度': 'mean',
        '海面気圧': 'mean'
    }).reset_index()
    
    # カラム名をわかりやすく整理 (例: tokyo_temp_max)
    df_daily.columns = ['day', f'{name}_temp_mean', f'{name}_temp_max', f'{name}_temp_min', f'{name}_hum', f'{name}_press']
    all_stations_daily.append(df_daily.set_index('day'))

# --- 3. 特徴量生成 ---
df_combined = pd.concat(all_stations_daily, axis=1)

# 全ての列の「前日データ(prev)」を作成
for col in df_combined.columns:
    df_combined[f'prev_{col}'] = df_combined[col].shift(1)

df_ml = df_combined.dropna()

# --- 4. 2つのモデルを学習・保存 ---
# 予測に使うヒント（昨日のデータ）
features = [col for col in df_ml.columns if 'prev_' in col]
X = df_ml[features]

# A. 最高気温モデル
y_max = df_ml['tokyo_temp_max']
model_max = LinearRegression()
model_max.fit(X, y_max)
with open('model_max.pkl', 'wb') as f:
    pickle.dump(model_max, f)
print(f"\n✅ 最高気温モデル保存完了 (R2: {model_max.score(X, y_max):.4f})")

# B. 最低気温モデル
y_min = df_ml['tokyo_temp_min']
model_min = LinearRegression()
model_min.fit(X, y_min)
with open('model_min.pkl', 'wb') as f:
    pickle.dump(model_min, f)
print(f"✅ 最低気温モデル保存完了 (R2: {model_min.score(X, y_min):.4f})")

print("\n✨ すべての工程が完了しました。フォルダ内に2つの .pkl ファイルがあることを確認してください。")