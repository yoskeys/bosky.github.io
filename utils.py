import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from datetime import timezone, timedelta

# --- 定数設定 ---
# 観測地点
STATIONS = {
    'tokyo': {'prec_no': 44, 'block_no': 47662},
    'kofu': {'prec_no': 49, 'block_no': 47638}
}

# 物理量を含む全項目
# ★ここが重要：CSVの列名と完全に一致させる必要があります
WEATHER_COLS = [
    'temp_mean', 'temp_max', 'temp_min', 
    'hum', 'press', 'precip', 'sun', 
    'dewpoint', 'theta_e', 
    'vpd',      # 追加: 飽和欠差
    'wind_u',   # 追加: 東西風
    'wind_v'    # 追加: 南北風
]

# 日本時間設定
JST = timezone(timedelta(hours=+9), 'JST')

# 風向を角度に変換するマッピング
WIND_DIR_MAP = {
    "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5,
    "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5,
    "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5,
    "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5,
    "静穏": 0
}

# --- 関数定義 ---

def get_wind_degrees(dir_str):
    """気象庁の風向テキスト(漢字)を角度(度)に変換"""
    return WIND_DIR_MAP.get(dir_str, np.nan)

def fetch_daily_data(date, prec_no, block_no):
    """
    今日や昨日のデータを取得し、MetPyで物理量(VPD, 風ベクトル等)を計算する
    """
    url = f"https://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no={prec_no}&block_no={block_no}&year={date.year}&month={date.month}&day={date.day}&view="
    
    try:
        r = requests.get(url, timeout=10)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.find_all('tr', class_='mtx')
        
        data = []
        for row in rows[2:]:
            cols = row.find_all('td')
            data.append([col.text.strip() for col in cols])
            
        df = pd.DataFrame(data)
        
        # 数値変換 (エラー値などはNaNへ)
        # 2:気圧, 3:降水, 4:気温, 7:湿度, 8:風速, 10:日照
        for col_idx in [2, 3, 4, 7, 8, 10]:
            df[col_idx] = pd.to_numeric(df[col_idx], errors='coerce')

        # --- 基本統計 ---
        t_mean = df[4].mean()
        hum_mean = df[7].mean()
        press_mean = df[2].mean()
        
        # --- 高度な物理量計算 (MetPy) ---
        t_obj = t_mean * units.degC
        rh_obj = (hum_mean / 100.0)
        p_obj = press_mean * units.hPa
        
        # 1. 露点 & 相当温位
        dewpoint_obj = mpcalc.dewpoint_from_relative_humidity(t_obj, rh_obj)
        theta_e_obj = mpcalc.equivalent_potential_temperature(p_obj, t_obj, dewpoint_obj)

        # 2. 飽和欠差 (VPD)
        e_sat_dew = mpcalc.saturation_vapor_pressure(dewpoint_obj)
        e_sat_temp = mpcalc.saturation_vapor_pressure(t_obj)
        vpd_obj = e_sat_temp - e_sat_dew

        # 3. 風ベクトル
        wind_speeds = df[8].values * units('m/s')
        wind_dirs = df[9].apply(get_wind_degrees).values * units.deg
        u_comp, v_comp = mpcalc.wind_components(wind_speeds, wind_dirs)
        
        u_mean = np.nanmean(u_comp.magnitude)
        v_mean = np.nanmean(v_comp.magnitude)

        return {
            'temp_mean': t_mean,
            'temp_max': df[4].max(),
            'temp_min': df[4].min(),
            'hum': hum_mean,
            'press': press_mean,
            'precip': df[3].fillna(0).sum(),
            'sun': df[10].fillna(0).sum(),
            'dewpoint': dewpoint_obj.magnitude,
            'theta_e': theta_e_obj.magnitude,
            'vpd': vpd_obj.magnitude, # ここも忘れずに！
            'wind_u': u_mean,
            'wind_v': v_mean
        }
        
    except Exception:
        return None

def build_input_vector(data_list):
    """リストデータをAI入力用の1行のベクトルに変換する"""
    v = []
    for day in data_list:
        for st_name in ['tokyo', 'kofu']:
            d = day[st_name]
            # 定義された全カラムを順番に抽出
            for col in WEATHER_COLS:
                v.append(d.get(col, 0)) # キーがない場合は0で埋める安全策
    return v

def generate_commentary(current_theta_e, prev_theta_e, pred_max, current_max):
    """
    解説文生成ロジック
    """
    # エネルギー判定
    theta_diff = current_theta_e - prev_theta_e
    
    if theta_diff >= 3.0:
        reason_energy = "暖かく湿ったエネルギーの高い空気が流入しており（相当温位の急上昇）、"
    elif theta_diff <= -3.0:
        reason_energy = "北からの乾燥した空気や寒気が流れ込んでいるため（相当温位の低下）、"
    elif theta_diff > 1.0:
        reason_energy = "大気のエネルギー状態は緩やかに上昇傾向にあり、"
    else:
        reason_energy = "大気の熱エネルギー状態は比較的安定していますが、"

    # 気温判定
    temp_diff = pred_max - current_max
    if temp_diff >= 2.0:
        reason_temp = "明日は今日よりも気温が「高く」なるでしょう。熱中症や脱水に注意が必要です。"
    elif temp_diff <= -2.0:
        reason_temp = "明日は今日よりも気温がグッと「低く」なりそうです。服装選びに気をつけてください。"
    elif temp_diff >= 0.5:
        reason_temp = "明日は今日よりわずかに気温が上がる見込みです。"
    elif temp_diff <= -0.5:
        reason_temp = "明日は今日よりわずかに気温が下がる見込みです。"
    else:
        reason_temp = "明日は今日とほぼ同じ気温で推移するでしょう。"

    return reason_energy + reason_temp