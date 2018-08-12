import pandas as pd

import time
start_time = time.time()

# input_file='/home/lia/Documents/the_project/dataset/top_50_movies/'
file_dir = '/home/lia/Documents/the_project/dataset/'
input_file = file_dir+"to_use/movies_sm_fixed.csv"

output_dir = file_dir+"helpfulness/"

df = pd.read_csv(input_file)
print(df['overall'].value_counts().sort_index())

# ANY REVIEWS THAT GOT UPVOTED BY >50% OF OTHER USERS
df_helpful = df.loc[(df['helpful_1'] > (df['helpful_2']/2)) & (df['helpful_1']>1)]
df_helpful.to_csv((output_dir+'helpful.csv'), sep=',', encoding='utf-8')

df_helpful['overall'].value_counts().sort_index().to_csv((output_dir+'helpful_overall.csv'), sep=',', encoding='utf-8')
df_helpful['asin'].value_counts().sort_index().to_csv((output_dir+'helpful_asin.csv'), sep=',', encoding='utf-8')

# ANY REVIEWS THAT WENT UNRECOGNIZED
df_zero = df[df['helpful_2'] == 0]
df_zero.to_csv((output_dir+'zero.csv'), sep=',', encoding='utf-8')

df_zero['overall'].value_counts().to_csv((output_dir+'zero_overall.csv'), sep=',', encoding='utf-8')
df_zero['asin'].value_counts().to_csv((output_dir+'zero_asin.csv'), sep=',', encoding='utf-8')
print(df_zero)

# ANY REVIEWS THAT GOT DOWNVOTED BY >50% OF OTHER USERS
df_unhelpful = df[~((df.index.isin(df_zero.index))|(df.index.isin(df_helpful.index)))]
df_helpful.to_csv((output_dir+'unhelpful.csv'), sep=',', encoding='utf-8')

df_unhelpful['overall'].value_counts().to_csv((output_dir+'unhelpful_overall.csv'), sep=',', encoding='utf-8')
df_unhelpful['asin'].value_counts().to_csv((output_dir+'unhelpful_asin.csv'), sep=',', encoding='utf-8')
print(df_unhelpful)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
