import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import time
import metpy.calc as mpcalc
from metpy.units import units

# --- 設定 ---
# 過去10年分のデータを取得します
start_date = datetime.date(2015, 1, 1)
# 昨日の日付まで（当日のデータはまだ確定していないため）
end_date = datetime.date.today() - datetime.timedelta(days=1)

# 観測地点（まずは東京と甲府で確実に動かしましょう）
STATIONS = {
    'tokyo': {'prec_no': 44, 'block_no': 47662},
    'kofu': {'prec_no': 49, 'block_no': 47638}
}

# 風向（文字）を角度（数字）に変換する辞書
WIND_DIR_MAP = {
    "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5,
    "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5,
    "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5,
    "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5,
    "静穏": 0
}

def get_wind_degrees(dir_str):
    # 辞書を使って文字を角度に変換します
    return WIND_DIR_MAP.get(dir_str, np.nan)

def fetch_daily_data_enhanced(date, prec_no, block_no):
    """
    1日分のデータを取得し、MetPyで高度な物理計算を行う関数
    """
    # 気象庁のURL生成
    url = f"https://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no={prec_no}&block_no={block_no}&year={date.year}&month={date.month}&day={date.day}&view="
    
    try:
        r = requests.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.find_all('tr', class_='mtx')
        
        # HTMLから表データを抽出
        data = []
        for row in rows[2:]: # ヘッダー行を飛ばす
            cols = row.find_all('td')
            data.append([col.text.strip() for col in cols])
        
        df = pd.DataFrame(data)
        
        # 必要な列を数値に変換（エラーがある場合はNaNにする）
        # 2:気圧, 3:降水, 4:気温, 7:湿度, 8:風速, 10:日照
        for col_idx in [2, 3, 4, 7, 8, 10]:
            df[col_idx] = pd.to_numeric(df[col_idx], errors='coerce')

        # 平均値を計算
        t_mean = df[4].mean()      # 気温
        hum_mean = df[7].mean()    # 湿度
        press_mean = df[2].mean()  # 気圧

        # データが欠けている日はスキップ（計算エラーを防ぐため）
        if pd.isna(t_mean) or pd.isna(hum_mean):
            return None

        # --- ここから気象学的計算 (MetPy) ---
        # 計算のために「単位」を付けます
        t_obj = t_mean * units.degC
        rh_obj = (hum_mean / 100.0)
        p_obj = press_mean * units.hPa
        
        # 1. 露点温度 & 相当温位の計算
        dewpoint_obj = mpcalc.dewpoint_from_relative_humidity(t_obj, rh_obj)
        theta_e_obj = mpcalc.equivalent_potential_temperature(p_obj, t_obj, dewpoint_obj)

        # 2. 飽和欠差 (VPD) の計算
        e_sat_dew = mpcalc.saturation_vapor_pressure(dewpoint_obj) # 今の水蒸気圧
        e_sat_temp = mpcalc.saturation_vapor_pressure(t_obj)       # 今の気温でのMAX水蒸気圧
        vpd_obj = e_sat_temp - e_sat_dew                           # その差（渇き具合）

        # 3. 風ベクトルの計算
        wind_speeds = df[8].values * units('m/s')
        wind_dirs = df[9].apply(get_wind_degrees).values * units.deg
        # 風向・風速から、東西成分(u)と南北成分(v)に分解
        u_comp, v_comp = mpcalc.wind_components(wind_speeds, wind_dirs)
        
        return {
            'temp_mean': t_mean,
            'temp_max': df[4].max(),
            'temp_min': df[4].min(),
            'hum': hum_mean,
            'press': press_mean,
            'precip': df[3].fillna(0).sum(),
            'sun': df[10].fillna(0).sum(),
            # ここが今回追加される「高度な物理量」です
            'dewpoint': dewpoint_obj.magnitude,
            'theta_e': theta_e_obj.magnitude,
            'vpd': vpd_obj.magnitude,
            'wind_u': np.nanmean(u_comp.magnitude),
            'wind_v': np.nanmean(v_comp.magnitude)
        }
    except Exception:
        return None

# --- メイン処理 ---
all_data = []
current_date = start_date

print(f"🚀 データ作成を開始します: {start_date} ～ {end_date}")
print("時間がかかります（約5〜10分）。コーヒーでも飲んでお待ちください☕")

start_time = time.time()

while current_date <= end_date:
    day_record = {'date': current_date}
    success = True
    
    for name, ids in STATIONS.items():
        # ここでデータを取得しに行きます
        res = fetch_daily_data_enhanced(current_date, ids['prec_no'], ids['block_no'])
        if res:
            for key, val in res.items():
                day_record[f"{name}_{key}"] = val
        else:
            success = False # どちらかの地点でデータが取れない日は使わない
    
    if success:
        all_data.append(day_record)
    
    # 進行状況を表示（50日ごと）
    if len(all_data) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"✅ {len(all_data)}日分完了... ({current_date}) - {elapsed:.0f}秒経過")
    
    # サーバー負荷軽減のための待機時間
    time.sleep(0.1) 
    current_date += datetime.timedelta(days=1)

# CSVファイルとして保存
df_final = pd.DataFrame(all_data)
df_final.to_csv('weather_database_enhanced.csv', index=False)

print("✨ 完了しました！ 'weather_database_enhanced.csv' が作成されました。")