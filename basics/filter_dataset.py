# get the most important attributes only

import pandas as pd
import csv

import time
start_time = time.time()

file_path = '/home/lia/Documents/the_project/dataset/output/'

df = pd.read_csv(file_path+'clean_data.csv')

new_df = df.loc[:,['asin', 'reviewText', 'overall', 'helpful']]

new_df['helpful_1'], new_df['helpful_2'] = new_df['helpful'].str.strip('[]').str.split(',', 1).str
new_df['helpful_1'] = new_df['helpful_1'].astype(int)
new_df['helpful_2'] = new_df['helpful_2'].astype(int)

new_df.drop('helpful', axis=1, inplace=True)

new_df.to_csv(file_path+'clean_data.csv', encoding="utf-8", sep=",", index=False)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
