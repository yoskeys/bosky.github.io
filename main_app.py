import streamlit as st

# よすきーの部屋風のヘッダーをAIアプリにも表示
st.markdown("""
    <style>
        .yoskey-header {
            text-align: center;
            color: #333;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
    <div class="yoskey-header">
        <h2>よすきーの部屋の天気予報</h2>
    </div>
    """, unsafe_allow_html=True)

# --- ここから下に、これまでの予測プログラムを続ける ---

import streamlit as st
import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# --- ページ設定 ---
st.set_page_config(page_title="天気予報")
st.title("🌡️ 7日間トレンド予報")
st.write("直近7日間の実況値を自動取得し、過去数年の同時期の傾向を『重み付け学習』して今日を予測します。")

# --- 設定：取得する地点 ---
STATIONS = {
    'tokyo': {'prec_no': 44, 'block_no': 47662},
    'kofu': {'prec_no': 49, 'block_no': 47638}
}

# --- 1. 気象庁から1日分のデータをスクレイピングする関数 ---
def fetch_daily_data(date, prec_no, block_no):
    url = f"https://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no={prec_no}&block_no={block_no}&year={date.year}&month={date.month}&day={date.day}&view="
    try:
        r = requests.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.find_all('tr', class_='mtx')
        data = []
        for row in rows[2:]:
            cols = row.find_all('td')
            data.append([col.text for col in cols])
        df = pd.DataFrame(data)
        # 数値変換（気温4, 湿度7, 気圧2, 降水3, 日照12）
        for col in [4, 7, 2, 3, 12]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return {
            'temp_mean': df[4].mean(), 'temp_max': df[4].max(), 'temp_min': df[4].min(),
            'hum': df[7].mean(), 'press': df[2].mean(), 'precip': df[3].sum(), 'sun': df[12].sum()
        }
    except:
        return None

# --- 2. 季節の重みを計算する関数 ---
def calculate_seasonal_weights(data_months, current_month):
    # 月の距離を計算（12月と1月は距離1）
    diff = np.abs(data_months - current_month)
    diff = np.where(diff > 6, 12 - diff, diff)
    # 距離が近いほど重みを大きく（1.0〜0.14の範囲）
    return 1.0 / (diff + 1)

# --- メイン処理 ---
if st.button('最新のトレンドを解析して予測開始'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # STEP 1: 直近7日間の実況値をリアルタイム取得
        recent_input_values = []
        # 1日前から7日前まで遡る
        target_dates = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(1, 8)]
        
        for i, date in enumerate(target_dates):
            status_text.text(f"📡 気象庁から実況データを取得中: {date} ({i+1}/7日分)")
            day_data = []
            for name, ids in STATIONS.items():
                res = fetch_daily_data(date, ids['prec_no'], ids['block_no'])
                if res:
                    day_data.extend([res['temp_mean'], res['temp_max'], res['temp_min'], res['hum'], res['press'], res['precip'], res['sun']])
                else:
                    raise Exception(f"{date}のデータ取得に失敗しました")
            recent_input_values.extend(day_data)
            progress_bar.progress((i + 1) / 7)
            time.sleep(0.2)

        # STEP 2: 学習データの準備（ラグ特徴量の作成）
        status_text.text("🧠 データベースから季節パターンを学習中...")
        df_all = pd.read_csv('weather_database.csv')
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_all = df_all.sort_values('date')

        # 特徴量（ヒント）の列名リストを作成
        features = []
        base_cols = ['temp_mean', 'temp_max', 'temp_min', 'hum', 'press', 'precip', 'sun']
        for lag in range(1, 8):
            for st_name in STATIONS.keys():
                for col in base_cols:
                    col_name = f'lag{lag}_{st_name}_{col}'
                    df_all[col_name] = df_all[f'{st_name}_{col}'].shift(lag)
                    features.append(col_name)

        # 欠損値（最初の7日分）を削除
        df_ml = df_all.dropna().copy()

        # STEP 3: 季節の重み付け学習
        current_month = datetime.date.today().month
        data_months = df_ml['date'].dt.month.values
        weights = calculate_seasonal_weights(data_months, current_month)

        # 今日の予測用入力データ
        input_df = pd.DataFrame([recent_input_values], columns=features)

        # 学習と予測
        final_results = {}
        for key, t_col in {'max': 'tokyo_temp_max', 'min': 'tokyo_temp_min'}.items():
            model = LinearRegression()
            model.fit(df_ml[features], df_ml[t_col], sample_weight=weights)
            final_results[key] = model.predict(input_df)[0]

        # STEP 4: 結果表示
        status_text.empty()
        st.success(f"解析完了！ 今の時期（{current_month}月）に最適化された予測です。")
        
        col1, col2 = st.columns(2)
        col1.metric("予想最高気温", f"{final_results['max']:.1f} ℃")
        col2.metric("予想最低気温", f"{final_results['min']:.1f} ℃")
        
        with st.expander("解析の詳細を確認"):
            st.write(f"学習データ件数: {len(df_ml)}件")
            st.write("使用したヒント: 直近7日間の気象推移（計98項目）")
            st.write("重み付け: 現在の月との近さに応じてデータの重要度を調整済み")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")