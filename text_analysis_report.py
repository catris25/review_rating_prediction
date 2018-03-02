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


input_file='/home/lia/Documents/the_project/dataset/top_10_movies/top_10.csv'

df = pd.read_csv(input_file)
new_df = tokenize_df(df)

print(new_df.describe())

data = new_df['sentence']

# print('Number of words {}'.format (Counter(new_df['word'])))
# print('Number of sentences {}'.format (Counter(new_df['sentence'])))

labels, values = zip(*Counter(data).items())

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels,rotation = "vertical")
plt.show()
