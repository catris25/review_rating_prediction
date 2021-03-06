# get the most reviewed films review Only
import pandas as pd
from langdetect import detect

import time
start_time = time.time()

# file_path = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/helpful.csv'
file_path = '/home/lia/Documents/the_project/dataset/movies_tv_reviews_sm.csv'


df = pd.read_csv(file_path)
s = df['asin'].value_counts().sort_values(ascending=False)
print(s)
#solution 1
# new_df = pd.DataFrame({'asin':s.index}).merge(df, how='left')
# print(new_df)

# output_file = '/home/lia/Documents/the_project/dataset/top_10_movies/top_1.csv'
# new_df.to_csv(output_file, encoding="utf-8", sep=",", index=False)

#solution 2
# new_df = df[df.asin.isin(df.asin.value_counts().head(20).index)]
# print(new_df)

#solution 3
# new_df = df.assign(rank=df.groupby('asin')['asin'].transform('size').rank(method='dense', ascending=False)).sort_values('rank').query("rank <= 20").drop('rank', 1)
# print(new_df)


time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
