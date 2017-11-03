import pandas as pd
import numpy as np

# input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/helpful_en.csv'
input_file = '/home/lia/Documents/the_project/dataset/top_50_movies/top_50.csv'

df = pd.read_csv(input_file)

avg = df.groupby('asin', as_index=False)['overall'].mean()
med = df.groupby('asin', as_index=False)['overall'].median()
print(df.groupby('asin')['overall'].describe())

# asin_list = df['asin'].value_counts().sort_values(ascending=False)
# print(asin_list)
#
# for al in asin_list:
#     df_film = np.where(df['asin']==al)
#     rating = df_film['overall'].mean()
#     print(rating)
