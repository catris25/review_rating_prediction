import pandas as pd
import numpy as np
import re, string
import sys

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# ---- PREPROCESSING STEP ----

# Punctuation removal, lowering capital letters, and tokenization
def removing_punct(df):
    df_removal = []
    for review in df['reviewText']:
        temp = review.lower()
        temp = re.sub("[^a-zA-Z]", " ", str(temp))
        temp = word_tokenize(temp)

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
    for review in df['reviewText']:
        stemmed_text = [ss.stem(word) for word in review]
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

    return df

def main():
    # input_file = '/home/lia/Documents/the_project/dataset/top_50_movies/top_50.csv'
    input_file = "/home/lia/Dropbox/output/additional_dataset/cleaned_data.csv"

    df = pd.read_csv(input_file)
    preprocessed_df = preprocess_data(df)

    output_file = "/home/lia/Dropbox/output/additional_dataset/preprocessed.csv"
    preprocessed_df.to_csv(output_file, index=False, sep=',')
    print("PREPROCESSING done")


if __name__ == "__main__":
    main()
