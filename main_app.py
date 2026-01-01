import streamlit as st
import pandas as pd
import datetime
import altair as alt
from sklearn.ensemble import RandomForestRegressor

# 自作モジュール
import utils 

# --- ページ設定 ---
st.set_page_config(page_title="Simple Weather AI", page_icon="🌤️")
st.title("よすきー天気")

# 日付取得
today = datetime.datetime.now(utils.JST).date()
tomorrow = today + datetime.timedelta(days=1)

# --- AIモデル構築 ---
@st.cache_resource
def load_smart_model():
    df_all = pd.read_csv('weather_database_enhanced.csv')
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    valid_features = []
    for lag in range(1, 8):
        for st_name in ['tokyo', 'kofu']:
            for col in utils.WEATHER_COLS:
                col_name = f'lag{lag}_{st_name}_{col}'
                if f'{st_name}_{col}' in df_all.columns:
                    df_all[col_name] = df_all[f'{st_name}_{col}'].shift(lag)
                    valid_features.append(col_name)
    
    df_ml = df_all.dropna(subset=valid_features).copy()
    
    models = {}
    for target_key, target_col in {'max': 'tokyo_temp_max', 'min': 'tokyo_temp_min'}.items():
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(df_ml[valid_features], df_ml[target_col])
        models[target_key] = model

    return models, valid_features

# --- メイン処理 ---
if st.button('予報を開始'):
    status_text = st.empty()
    
    try:
        # ① データ取得
        status_text.text("📡 データを解析中...")
        recent_actual_data = [] 
        target_dates = [(today - datetime.timedelta(days=i)) for i in range(1, 8)]
        
        for date in reversed(target_dates):
            day_results = {}
            for name, ids in utils.STATIONS.items():
                day_results[name] = utils.fetch_daily_data(date, ids['prec_no'], ids['block_no'])
            recent_actual_data.insert(0, day_results)

        # ② 予測実行
        models, valid_features = load_smart_model()
        
        # 今日の予測
        input_values = utils.build_input_vector(recent_actual_data) 
        input_today_df = pd.DataFrame([input_values], columns=valid_features)
        
        preds_today = {}
        for key in ['max', 'min']:
            preds_today[key] = models[key].predict(input_today_df)[0]

        # 明日の予測
        predicted_record = {}
        for st_name in utils.STATIONS.keys():
            t_mean = (preds_today['max'] + preds_today['min']) / 2
            prev = recent_actual_data[0][st_name]
            predicted_record[st_name] = {
                'temp_mean': t_mean, 'temp_max': preds_today['max'], 'temp_min': preds_today['min'],
                'hum': prev['hum'], 'press': prev['press'], 'precip': 0, 'sun': prev['sun'],
                'dewpoint': prev['dewpoint'], 'theta_e': prev['theta_e'],
                'vpd': prev['vpd'], 'wind_u': prev['wind_u'], 'wind_v': prev['wind_v']
            }
        
        future_input_list = [predicted_record] + recent_actual_data[:-1]
        input_tomorrow_values = utils.build_input_vector(future_input_list)
        input_tomorrow_df = pd.DataFrame([input_tomorrow_values], columns=valid_features)
        
        preds_tomorrow = {}
        for key in ['max', 'min']:
            preds_tomorrow[key] = models[key].predict(input_tomorrow_df)[0]

        status_text.empty()

        # --- UI表示 ---

        # 昨日のデータを取得（比較用）
        yesterday_data = recent_actual_data[0]['tokyo']

        # 1. 今日の気温 (前日比を追加)
        st.subheader(f"今日 {today.strftime('%m/%d')}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "最高気温", 
                f"{preds_today['max']:.1f}℃", 
                delta=f"{preds_today['max'] - yesterday_data['temp_max']:.1f}℃"
            )
        with col2:
            st.metric(
                "最低気温", 
                f"{preds_today['min']:.1f}℃", 
                delta=f"{preds_today['min'] - yesterday_data['temp_min']:.1f}℃"
            )

        # 2. 明日の気温 (今日比)
        st.subheader(f"明日 {tomorrow.strftime('%m/%d')}")
        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                "最高気温", 
                f"{preds_tomorrow['max']:.1f}℃", 
                delta=f"{preds_tomorrow['max'] - preds_today['max']:.1f}℃"
            )
        with col4:
            st.metric(
                "最低気温", 
                f"{preds_tomorrow['min']:.1f}℃", 
                delta=f"{preds_tomorrow['min'] - preds_today['min']:.1f}℃"
            )

        # 3. AI解説
        latest_data = recent_actual_data[0]['tokyo']
        prev_data = recent_actual_data[1]['tokyo']
        commentary = utils.generate_commentary(
            latest_data['theta_e'], prev_data['theta_e'], 
            preds_tomorrow['max'], preds_today['max']
        )
        st.markdown("---")
        st.markdown(f"**🤖 予報の根拠**\n\n{commentary}")

        # 4. 推移グラフ
        st.markdown("---")
        st.caption("過去7日間の気温推移")
        
        summary = []
        for i, date in enumerate(target_dates):
            d = recent_actual_data[i]['tokyo']
            summary.append({
                "日付": date.strftime('%m/%d'), 
                "最高気温": d['temp_max'], 
                "最低気温": d['temp_min']
            })
        df_summary = pd.DataFrame(summary)

        base = alt.Chart(df_summary).encode(x=alt.X('日付', sort=None))
        line_max = base.mark_line(color='#ff6b6b', point=True).encode(
            y=alt.Y('最高気温', scale=alt.Scale(zero=False), title='気温 (℃)'),
            tooltip=['日付', '最高気温']
        )
        line_min = base.mark_line(color='#4d96ff', point=True).encode(
            y=alt.Y('最低気温', scale=alt.Scale(zero=False)),
            tooltip=['日付', '最低気温']
        )
        st.altair_chart((line_max + line_min).properties(height=250), use_container_width=True)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")