# get the most important attributes only

import pandas as pd
import csv

import time
start_time = time.time()

file_path = '/home/lia/Documents/the_project/dataset/Movies_and_TV_review.csv'

df = pd.read_csv(file_path)

new_df = df.loc[:,['asin', 'reviewText', 'overall', 'helpful']]
print(new_df.head(10))

new_df.to_csv("densed_reviews.csv", encoding="utf-8", sep=",", index=False)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
