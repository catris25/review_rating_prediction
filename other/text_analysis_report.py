import numpy as np
import pandas as pd
import re, math
from collections import Counter

import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer, TweetTokenizer

# REMOVE ALL PUNCTUATIONS AND THEN TOKENIZE THE TEXT
def tokenize_df(df):
    df_token = []
    for review in df['reviewText']:
        temp = review
        sent_length = len(sent_tokenize(temp))
        temp = re.sub("[^a-zA-Z']", " ", str(review))
        temp = temp.replace("'", "")
        temp = temp.lower()
        word_length = len(word_tokenize(temp))

        df_token.append({'reviewText': temp, 'word':word_length, 'sentence':sent_length})

    df_token = pd.DataFrame(df_token)

    return df_token


input_file='/home/lia/Documents/the_project/dataset/to_use/current/top_30.csv'
# input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/6.csv'

df = pd.read_csv(input_file)
new_df = tokenize_df(df)

print(new_df.describe())
print(new_df.head(10))
# data = new_df['word']
#
# plt.hist(data, bins=200)
# plt.show()

# def outliers_z_score(ys):
#     threshold = 3
#
#     mean_y = np.mean(ys)
#     stdev_y = np.std(ys)
#     z_scores = [(y - mean_y) / stdev_y for y in ys]
#     return np.where(np.abs(z_scores) > threshold)
#
# oz = outliers_z_score(data)
# print(oz)

# print('Number of words {}'.format (Counter(new_df['word'])))
# print('Number of sentences {}'.format (Counter(new_df['sentence'])))

# labels, values = zip(*Counter(data).items())
#
# indexes = np.arange(len(labels))
# width = 1
#
# plt.bar(indexes, values, width)
# plt.xticks(indexes + width * 0.5, labels,rotation = "vertical")
# plt.show()

# for w in new_df['word']:
#     if w<=10:
#         print(w)


too_long = df.loc[new_df['word'] >= 1000, 'reviewText']
too_short = df.loc[new_df['word'] <= 10, 'reviewText']
print('too long:', len(too_long))
print('too short:', len(too_short))

df['word'] = new_df['word']

del_id = too_long.index.append(too_short.index)
temp_df = df.drop(df.index[[del_id]])

print(temp_df.head(10))
# 
# temp_df.to_csv('/home/lia/Documents/the_project/dataset/top_10_movies/top_10_clean.csv')
