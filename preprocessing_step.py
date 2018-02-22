import pandas as pd
import numpy as np
import re, string
import sys

import nltk
from nltk import bigrams, trigrams, FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.classify import NaiveBayesClassifier

# ---- PREPROCESSING STEP ----
# Punctuation removal, lowering capital letters, and tokenization
def removing_punct(df):
    df_removal = []
    tw = TweetTokenizer()
    for review in df['reviewText']:
        temp = re.sub("[^a-zA-Z']", " ", str(review))
        temp = temp.replace("'", "")
        temp = temp.lower()
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

    for review in df['reviewText']:
        filtered_text = [word for word in review if not word in stop_words]
        df_filtering.append(filtered_text)

    df['reviewText'] = df_filtering
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

def preprocess_data(input_df):
    df = input_df

    print(df['overall'].value_counts().sort_index())
    print(df[['reviewText', 'overall']].head(5))

    df = removing_punct(df)
    df = stop_words_removal(df)
    df = stemming(df)

    output_file = '/home/lia/Documents/the_project/dataset/output/temp.csv'
    df.to_csv(output_file, index=False, sep=',')
    df = pd.read_csv(output_file)

    return df

def main():
    input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/10percent/6.csv'

    df = pd.read_csv(input_file)
    preprocessed_df = preprocess_data(df)

    output_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'
    preprocessed_df.to_csv(output_file, index=False, sep=',')
    print("PREPROCESSING done")


if __name__ == "__main__":
    main()
