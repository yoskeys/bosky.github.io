import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. 2024年のデータを読み込む
df = pd.read_csv('weather_data_2024_full.csv')

# 2. 11月・12月・1月だけを抽出 (冬モデル)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df_seasonal = df[df['month'].isin([11, 12, 1])].copy()

# 3. 学習用のヒント（14項目）を作成
features = []
base_cols = ['temp_mean', 'temp_max', 'temp_min', 'hum', 'press', 'precip', 'sun']
stations = ['tokyo', 'kofu']

for st in stations:
    for col in base_cols:
        col_name = f'{st}_{col}'
        df_seasonal[f'prev_{col_name}'] = df_seasonal[col_name].shift(1)
        features.append(f'prev_{col_name}')

df_ml = df_seasonal.dropna()

# 4. 学習と保存 (これでモデルが14項目を覚えます)
target_cols = {'max': 'tokyo_temp_max', 'min': 'tokyo_temp_min'}

for key, t_col in target_cols.items():
    X = df_ml[features]
    y = df_ml[t_col]
    model = LinearRegression().fit(X, y)
    with open(f'model_{key}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ {key}気温予測モデル（冬版・14項目）完成！")