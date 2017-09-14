import pandas as pd

import time
start_time = time.time()

input_file='/home/lia/Documents/the_project/dataset/to_use/top_30_music.csv'
# input_file = '/home/lia/Documents/the_project/dataset/test.csv'

df = pd.read_csv(input_file)

df['helpful_1'], df['helpful_2'] = df['helpful'].str.strip('[]').str.split(',', 1).str
df['helpful_1'] = df['helpful_1'].astype(int)
df['helpful_2'] = df['helpful_2'].astype(int)

df.drop('helpful', axis=1, inplace=True)

output_file = '/home/lia/Documents/the_project/dataset/to_use/new_top_30_music.csv'
df.to_csv(output_file, encoding="utf-8", sep=",", index=False)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
