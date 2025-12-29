import streamlit as st
import datetime
from datetime import timezone, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# --- 1. ページ設定とタイトル ---
st.set_page_config(page_title="よすきー天気予報", page_icon="🌤️")
st.title("よすきー天気予報")
st.markdown("""
**過去10年のデータを学習したビッグデータモデルです。**
直近7日間のトレンドから今日を予測し、その結果を元に明日まで見通します。
""")

# 観測地点の設定
STATIONS = {
    'tokyo': {'prec_no': 44, 'block_no': 47662},
    'kofu': {'prec_no': 49, 'block_no': 47638}
}

# --- 2. 関数定義 ---

def fetch_daily_data(date, prec_no, block_no):
    """気象庁から指定日のデータをスクレイピング"""
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
        # 必要な列を数値に変換
        for col in [4, 7, 2, 3, 12]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return {
            'temp_mean': df[4].mean(), 'temp_max': df[4].max(), 'temp_min': df[4].min(),
            'hum': df[7].mean(), 'press': df[2].mean(), 'precip': df[3].sum(), 'sun': df[12].sum()
        }
    except:
        return None

def calculate_seasonal_weights(data_months, current_month):
    """現在の月との近さに応じてデータの重みを計算"""
    diff = np.abs(data_months - current_month)
    diff = np.where(diff > 6, 12 - diff, diff)
    return 1.0 / (diff + 1)

def build_input_vector(data_list):
    """7日分のデータを1つの入力ベクトルに変換"""
    v = []
    for day in data_list: # 1日前〜7日前
        for st_name in ['tokyo', 'kofu']:
            d = day[st_name]
            v.extend([d['temp_mean'], d['temp_max'], d['temp_min'], d['hum'], d['press'], d['precip'], d['sun']])
    return v

# --- 3. メイン処理 (予測開始) ---

if st.button('最新トレンドを解析して未来を予測する'):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 日付の計算
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.datetime.now(JST).date() 
    tomorrow = today + datetime.timedelta(days=1)
    
    try:
        # ① 直近7日間の実況値を取得
        recent_actual_data = [] 
        target_dates = [(today - datetime.timedelta(days=i)) for i in range(1, 8)]
        
        for i, date in enumerate(reversed(target_dates)): # 古い順(7日前)から取得
            status_text.text(f"📡 気象庁より実況データを取得中: {date}")
            day_results = {}
            for name, ids in STATIONS.items():
                day_results[name] = fetch_daily_data(date, ids['prec_no'], ids['block_no'])
            recent_actual_data.insert(0, day_results) # 常に先頭に入れ、[1日前, 2日前...7日前] の順にする
            progress_bar.progress((i + 1) / 7)
            time.sleep(0.1)

        # ② 10年分データを用いた学習
        status_text.text("🧠 過去10年の歴史を学習中...")
        df_all = pd.read_csv('weather_database.csv')
        df_all['date'] = pd.to_datetime(df_all['date'])
        
        # ラグ特徴量(過去7日分)の作成
        features = []
        for lag in range(1, 8):
            for st_name in ['tokyo', 'kofu']:
                for col in ['temp_mean', 'temp_max', 'temp_min', 'hum', 'press', 'precip', 'sun']:
                    col_name = f'lag{lag}_{st_name}_{col}'
                    df_all[col_name] = df_all[f'{st_name}_{col}'].shift(lag)
                    features.append(col_name)
        
        df_ml = df_all.dropna().copy()
        current_month = today.month
        weights = calculate_seasonal_weights(df_ml['date'].dt.month.values, current_month)

        # ③ 今日の予測実行
        status_text.text("🧪 今日を解析中...")
        input_today = pd.DataFrame([build_input_vector(recent_actual_data)], columns=features)
        
        preds_today = {}
        models = {}
        for key, t_col in {'max': 'tokyo_temp_max', 'min': 'tokyo_temp_min'}.items():
            model = LinearRegression().fit(df_ml[features], df_ml[t_col], sample_weight=weights)
            preds_today[key] = model.predict(input_today)[0]
            models[key] = model

        # ④ 明日の予測 (2段階予測)
        status_text.text("🚀 明日を計算中...")
        predicted_today_record = {}
        for st_name in STATIONS.keys():
            t_mean = (preds_today['max'] + preds_today['min']) / 2
            prev_day = recent_actual_data[0][st_name] # 昨日の実況値を流用
            predicted_today_record[st_name] = {
                'temp_mean': t_mean, 'temp_max': preds_today['max'], 'temp_min': preds_today['min'],
                'hum': prev_day['hum'], 'press': prev_day['press'], 'precip': 0, 'sun': prev_day['sun']
            }
        
        # 1日前を予測値、2-7日前を実況値にする
        future_input_list = [predicted_today_record] + recent_actual_data[:-1]
        input_tomorrow = pd.DataFrame([build_input_vector(future_input_list)], columns=features)
        
        preds_tomorrow = {}
        for key in ['max', 'min']:
            preds_tomorrow[key] = models[key].predict(input_tomorrow)[0]

        # --- 4. 結果表示 ---
        status_text.empty()
        progress_bar.empty()
        st.success("全ての解析が完了しました！")

        # A. 実績データの掲示 (小数点1桁、日照時間なし)
        st.markdown("---")
        st.subheader("直近7日間の観測 (東京)")
        st.write("AIが予測の根拠とした実際の気象推移です。")
        actual_summary = []
        for i, date in enumerate(target_dates):
            d = recent_actual_data[i]['tokyo']
            actual_summary.append({
                "日付": date.strftime('%m/%d'),
                "最高気温 (℃)": f"{d['temp_max']:.1f}",
                "最低気温 (℃)": f"{d['temp_min']:.1f}",
                "平均湿度 (%)": int(d['hum'])
            })
        st.table(pd.DataFrame(actual_summary))

        # B. 予測結果の掲示 (日付入りタイトル)
        st.markdown("---")
        t_col, m_col = st.columns(2)
        with t_col:
            st.subheader(f"今日 ({today.strftime('%m/%d')}) の予報")
            st.metric("最高気温", f"{preds_today['max']:.1f} ℃")
            st.metric("最低気温", f"{preds_today['min']:.1f} ℃")
            
        with m_col:
            st.subheader(f"明日 ({tomorrow.strftime('%m/%d')}) の予報")
            st.metric("最高気温", f"{preds_tomorrow['max']:.1f} ℃", delta=f"{preds_tomorrow['max'] - preds_today['max']:.1f} ℃")
            st.metric("最低気温", f"{preds_tomorrow['min']:.1f} ℃", delta=f"{preds_tomorrow['min'] - preds_today['min']:.1f} ℃")

        st.info(f"学習データ数: {len(df_ml)}件 / 重み付け対象月: {current_month}月")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")