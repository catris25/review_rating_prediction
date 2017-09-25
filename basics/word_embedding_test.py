import gensim
from tqdm import tqdm
import re
import nltk
from nltk import PunktSentenceTokenizer, word_tokenize
import pandas as pd
import numpy as np


def removing_punct(df):
    df_removal = []
    ps_token = PunktSentenceTokenizer()
    for review in df['reviewText']:
        temp = re.sub("[^a-zA-Z']", " ", str(review))
        temp = temp.replace("'", "")
        temp = temp.lower()
        temp = word_tokenize(temp)

        df_removal.append(temp)

    df['reviewText'] = df_removal
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    return df

def sep_to_x_y(df):
    reviews = df['reviewText']
    X_data = [(',').join(review) for review in reviews ]
    y_data = df['overall']

    return X_data, y_data

input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/20percent/4.csv'
orig_df = pd.read_csv(input_file)

print(orig_df['overall'].value_counts().sort_index())
print(orig_df.head(5))

df = removing_punct(orig_df)

X_data, y_data = sep_to_x_y(df)

# let X be a list of tokenized texts (i.e. list of lists of tokens)
# model = gensim.models.Word2Vec(X_data)
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))

w2v = gensim.models.Word2Vec(size = 200, min_count=10)
w2v.build_vocab([x['reviewText'] for x in tqdm(X_data)])
w2v.train([x['reviewText'] for x in tqdm(X_data)])

print(w2v['sad'])
