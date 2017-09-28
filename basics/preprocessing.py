import pandas as pd
import numpy as np
import re, string
import sys

import nltk

from nltk import bigrams, trigrams, FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer, TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.classify import NaiveBayesClassifier

# ---- PREPROCESSING STEP ----
# STEP 1. punctuation removal, lowering capital letters, and tokenization
def removing_punct(df):
    df_removal = []
    ps_token = PunktSentenceTokenizer()
    tw = TweetTokenizer()
    for review in df['reviewText']:
        temp = re.sub("[^a-zA-Z']", " ", str(review))
        temp = temp.replace("'", "")
        temp = temp.lower()
        # temp = word_tokenize(temp)
        temp = tw.tokenize(temp)

        df_removal.append(temp)

    df['reviewText'] = df_removal
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    return df

# Stop words removal/filtering
def stop_words_removal(df):
    df_filtering = []
    sw = set(stopwords.words("english"))
    stop_words = sw.union(["film", "movie", "movies", "films"])
    # stop_words = sw

    for review in df['reviewText']:
        filtered_text = [word for word in review if not word in stop_words]
        df_filtering.append(filtered_text)

    df['reviewText'] = df_filtering
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    return df

    #STEP 4. Lemmatization

def lemmatize(df):
    df_lemma = []
    wnl = WordNetLemmatizer()

    for review in df['reviewText']:
        lemmatized_text = [wnl.lemmatize(word, 'v') for word in review]
        lemmatized_text = [wnl.lemmatize(word, 'n') for word in lemmatized_text]
        lemmatized_text = [wnl.lemmatize(word, 'a') for word in lemmatized_text]
        df_lemma.append(lemmatized_text)

    df['reviewText'] = df_lemma
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    return df

# Stemming
def stemming(df):
    df_stemming = []
    ss = SnowballStemmer('english')
    ps = PorterStemmer()
    for review in df['reviewText']:
        stemmed_text = [ps.stem(word) for word in review]
        df_stemming.append(stemmed_text)

    df['reviewText'] = df_stemming
    print(df['reviewText'].head(5))

    return df

# input_file = '/home/lia/Documents/the_project/dataset/top_30_movies/helpful/10percent/3.csv'
input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/10percent/0.csv'
# input_file = '/home/lia/Documents/the_project/dataset/top_50_movies/helpful.csv'
# input_file = "/home/lia/Documents/the_project/dataset/clean_airline_sentiments.csv"
# input_file = "/home/lia/Documents/the_project/dataset/musical_inst/helpful.csv"
# input_file = '/home/lia/Documents/the_project/dataset/top_30_movies/new_set/helpful_en.csv'

orig_df = pd.read_csv(input_file)
print(orig_df['overall'].value_counts().sort_index())

df = removing_punct(orig_df)
df = stop_words_removal(df)
df = lemmatize(df)
df = stemming(df)

output_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'
df.to_csv(output_file, index=False, sep=',')
