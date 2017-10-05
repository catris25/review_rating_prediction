import pandas as pd
import numpy as np

from langdetect import detect

local_file = '/home/lia/Documents/the_project/dataset/musical_inst/'
# local_file ='/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/20percent/0.csv'
# local_file = '/home/lia/Documents/the_project/dataset/top_30_movies/helpful/10percent/3.csv'
df = pd.read_csv(local_file+'helpful_en.csv')

for i, row in df.iterrows():
    review = row['reviewText']
    lang = detect(str(review))
    if lang!='en':
        print(lang)
        print(review)
        df.drop(i, inplace=True)
        print("not eligible")
        # break

# new_df = df[detect(str(df['reviewText']))!='en']
# new_df = df[] [detect(str(df['reviewText']))!='en']

# print(df)
# df.to_csv((local_file+'helpful_en.csv'),encoding="utf-8", sep=",")

# 10% = 0
# 20% = 4,5
# 30% = 21,25
