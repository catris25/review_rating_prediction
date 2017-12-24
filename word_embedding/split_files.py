import pandas as pd
import numpy as np
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer, TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer


def removing_punct(df):
    df_removal = []
    ps_token = PunktSentenceTokenizer()
    for review in df['reviewText']:
        temp = re.sub("[^a-zA-Z']", " ", str(review))
        temp = temp.replace("'", "")
        temp = re.sub("\s\s+", " ", temp)
        temp = temp.lower()
        df_removal.append(temp)

    df['reviewText'] = df_removal
    print(df['reviewText'].head(5))
    # print(sum([len(r) for r in df['reviewText']]))

    return df


input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/helpful_en.csv'

df = pd.read_csv(input_file)

df = df[['reviewText', 'overall']]
print(df['overall'].value_counts())

pos = df[df['overall']==5]
pos = pos[['reviewText']]
pos = removing_punct(pos)

neg = df[df['overall']==1]
neg = neg[['reviewText']]
neg = removing_punct(neg)

output_file = '/home/lia/Documents/the_project/dataset/output/word_embedding/pos.txt'
pos.to_csv(output_file, index=False, sep=',')

output_file = '/home/lia/Documents/the_project/dataset/output/word_embedding/neg.txt'
neg.to_csv(output_file, index=False, sep=',')
