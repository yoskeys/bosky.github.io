import streamlit as st
import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# --- ページ設定 ---
st.set_page_config(page_title="よすきー気象予報", page_icon="🌤️")
st.title("AI予報")
st.write("最新7日間のトレンドから今日を予測し、その結果を元に明日まで見通します。")

STATIONS = {
    'tokyo': {'prec_no': 44, 'block_no': 47662},
    'kofu': {'prec_no': 49, 'block_no': 47638}
}

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
        for col in [4, 7, 2, 3, 12]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return {
            'temp_mean': df[4].mean(), 'temp_max': df[4].max(), 'temp_min': df[4].min(),
            'hum': df[7].mean(), 'press': df[2].mean(), 'precip': df[3].sum(), 'sun': df[12].sum()
        }
    except:
        return None

def calculate_seasonal_weights(data_months, current_month):
    diff = np.abs(data_months - current_month)
    diff = np.where(diff > 6, 12 - diff, diff)
    return 1.0 / (diff + 1)

if st.button('未来（明日）まで予測する'):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # 1. 直近7日間の実況値を取得
        recent_actual_data = [] # 1日前〜7日前のリスト（要素は辞書）
        target_dates = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(1, 8)]
        
        for i, date in enumerate(target_dates):
            status_text.text(f"📡 実況取得中: {date}")
            day_results = {}
            for name, ids in STATIONS.items():
                day_results[name] = fetch_daily_data(date, ids['prec_no'], ids['block_no'])
            recent_actual_data.append(day_results)
            progress_bar.progress((i + 1) / 7)
            time.sleep(0.1)

        # 特徴量ベクトル（入力データ）の作成
        def build_input_vector(data_list):
            v = []
            for day in data_list: # 1日前〜7日前
                for st_name in ['tokyo', 'kofu']:
                    d = day[st_name]
                    v.extend([d['temp_mean'], d['temp_max'], d['temp_min'], d['hum'], d['press'], d['precip'], d['sun']])
            return v

        # 2. 学習準備
        df_all = pd.read_csv('weather_database.csv')
        df_all['date'] = pd.to_datetime(df_all['date'])
        features = []
        for lag in range(1, 8):
            for st_name in ['tokyo', 'kofu']:
                for col in ['temp_mean', 'temp_max', 'temp_min', 'hum', 'press', 'precip', 'sun']:
                    col_name = f'lag{lag}_{st_name}_{col}'
                    df_all[col_name] = df_all[f'{st_name}_{col}'].shift(lag)
                    features.append(col_name)
        df_ml = df_all.dropna().copy()
        
        current_month = datetime.date.today().month
        weights = calculate_seasonal_weights(df_ml['date'].dt.month.values, current_month)

        # 3. 今日の予測実行
        status_text.text("🧠 今日の天気を解析中...")
        input_today = pd.DataFrame([build_input_vector(recent_actual_data)], columns=features)
        
        preds_today = {}
        models = {}
        for key, t_col in {'max': 'tokyo_temp_max', 'min': 'tokyo_temp_min'}.items():
            model = LinearRegression().fit(df_ml[features], df_ml[t_col], sample_weight=weights)
            preds_today[key] = model.predict(input_today)[0]
            models[key] = model # 明日のために保存

        # 4. 明日の予測（未来を広げる）
        status_text.text("🚀 今日の予測を元に、明日を計算中...")
        
        # 「今日」のダミーデータを作成（予測値を利用し、他は平均値などで補完）
        # ※本来は湿度なども予測すべきですが、まずは気温をスライドさせます
        predicted_today_record = {}
        for st_name in STATIONS.keys():
            # 今日の平均は最高と最低の間とする
            t_mean = (preds_today['max'] + preds_today['min']) / 2
            # 他の項目は「昨日」の値を一旦流用（簡易的なスライド）
            prev_day = recent_actual_data[0][st_name]
            predicted_today_record[st_name] = {
                'temp_mean': t_mean, 'temp_max': preds_today['max'], 'temp_min': preds_today['min'],
                'hum': prev_day['hum'], 'press': prev_day['press'], 'precip': 0, 'sun': prev_day['sun']
            }
        
        # 未来へスライド： 1日前を「予測した今日」にし、2〜7日前をこれまでの1〜6日前にする
        future_input_list = [predicted_today_record] + recent_actual_data[:-1]
        input_tomorrow = pd.DataFrame([build_input_vector(future_input_list)], columns=features)
        
        preds_tomorrow = {}
        for key in ['max', 'min']:
            preds_tomorrow[key] = models[key].predict(input_tomorrow)[0]

        # 5. 結果表示
        status_text.empty()
        st.success("今日と明日の予測が完了しました！")
        
        t_col, m_col = st.columns(2)
        with t_col:
            st.subheader("📌 今日の予報")
            st.metric("最高気温", f"{preds_today['max']:.1f} ℃")
            st.metric("最低気温", f"{preds_today['min']:.1f} ℃")
            
        with m_col:
            st.subheader("📅 明日の予報")
            st.metric("最高気温", f"{preds_tomorrow['max']:.1f} ℃", delta=f"{preds_tomorrow['max'] - preds_today['max']:.1f} ℃")
            st.metric("最低気温", f"{preds_tomorrow['min']:.1f} ℃", delta=f"{preds_tomorrow['min'] - preds_today['min']:.1f} ℃")

    except Exception as e:
        st.error(f"予測エラー: {e}")