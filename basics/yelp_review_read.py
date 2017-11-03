import pandas as pd
import numpy as np

from langdetect import detect

input_file = '/home/lia/Documents/the_project/dataset/yelp/yelp_hotel.csv'

df = pd.read_csv(input_file)

new_df = df[['reviews.text', 'reviews.rating']]

new_df.columns = ['reviewText', 'overall']
print(new_df.head(100))
# print(new_df['overall'].value_counts())


new_df = new_df.loc[(new_df['overall'] == 1) | (new_df['overall'] == 2) | (new_df['overall'] == 3)
| (new_df['overall'] == 4) | (new_df['overall'] == 5)]

(new_df.reviewText).drop_duplicates()
print(new_df['overall'].value_counts())

# for i, row in new_df.iterrows():
#     review = row['reviewText']
#     lang = detect(str(review))
#     if lang!='en':
#         print(lang)
#         print(review)
#         new_df.drop(i, inplace=True)
#
# new_df.to_csv(('/home/lia/Documents/the_project/dataset/yelp/clean_data.csv'), sep=",", encoding="utf-8", index=False)
