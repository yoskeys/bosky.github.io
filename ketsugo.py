import pandas as pd

# 1. 二つのCSVを読み込む
df1 = pd.read_csv('weather_data_2024_full.csv')
df2 = pd.read_csv('weather_data_extra.csv')

# 2. 縦に連結する
df_combined = pd.concat([df1, df2], ignore_index=True)

# 3. 日付型に変換して、古い順から新しい順に並べ替える
df_combined['date'] = pd.to_datetime(df_combined['date'])
df_combined = df_combined.sort_values('date').reset_index(drop=True)

# 4. 「完成版データベース」として保存
df_combined.to_csv('weather_database.csv', index=False)

print("✨ データの統合が完了しました！ 'weather_database.csv' を作成しました。")