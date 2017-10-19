import pandas as pd

local_dir = '/home/lia/Documents/the_project/dataset/top_10_movies/'

df = pd.read_csv(local_dir+'top_10.csv')

df['asin'].astype(str)
'"' + df['asin'] + '"'

asin_list = df['asin'].value_counts()

print(asin_list)
asin_list.to_csv((local_dir+'asin_list.csv'), encoding="utf-8", sep=",")
