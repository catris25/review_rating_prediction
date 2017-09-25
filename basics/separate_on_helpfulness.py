import pandas as pd

import time
start_time = time.time()

input_file='/home/lia/Documents/the_project/dataset/top_50_movies/'
# input_file = '/home/lia/Documents/the_project/dataset/musical_inst/'
# input_file = '/home/lia/Documents/the_project/dataset/test.csv'

df = pd.read_csv(input_file+'top_50.csv')
print(df['overall'].value_counts().sort_index())
# any review which is upvoted by >50% other users
df_helpful = df.loc[(df['helpful_1'] > (df['helpful_2']/2)) & (df['helpful_1']>1)]
df_helpful.to_csv((input_file+'helpful.csv'), sep=',', encoding='utf-8')

# df_helpful['overall'].value_counts().sort_index().to_csv((input_file+'full/helpful_overall.csv'), sep=',', encoding='utf-8')
# df_helpful['asin'].value_counts().sort_index().to_csv((input_file+'full/helpful_asin.csv'), sep=',', encoding='utf-8')


# any review which goes unrecognized
# df_zero = df[df['helpful_2'] == 0]
# df_zero.to_csv('zero.csv', sep=',', encoding='utf-8')
#
# df_zero['overall'].value_counts().to_csv('zero_overall.csv', sep=',', encoding='utf-8')
# df_zero['asin'].value_counts().to_csv('zero_asin.csv', sep=',', encoding='utf-8')
# print(df_zero)

# any review which is downvoted by >= 50% other users
# df_unhelpful = df[~((df.index.isin(df_zero.index))|(df.index.isin(df_helpful.index)))]
# df_helpful.to_csv('unhelpful.csv', sep=',', encoding='utf-8')
#
# df_unhelpful['overall'].value_counts().to_csv('unhelpful_overall.csv', sep=',', encoding='utf-8')
# df_unhelpful['asin'].value_counts().to_csv('unhelpful_asin.csv', sep=',', encoding='utf-8')
# print(df_unhelpful)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
