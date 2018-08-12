# LIA RISTIANA 2018
# This file deals with the cleaning of dataset process of the following steps:
# > get important columns only (done outside this file)
# > fix the helpful columns (done outside this file)
# > separate on helpfulness (done outside this file)
# Then feed the helpful reviews to this program as follows:
# 1. get n top film reviews
# 2. remove non-english reviews
# 3. remove too short reviews

# notes
# what to viusalize in text classification in a scatter plot?

import pandas as pd
import numpy as np

import re
from nltk.tokenize import word_tokenize
from langdetect import detect

# GET n TOP FILMS
def get_top_movies(df, n_movies):
    print("GETTING %d MOST REVIEWED MOVIES"%n_movies)

    # MODIFY THE LIST OF FILMS NEEDED TO FILTER
    df_list = pd.read_csv('/home/lia/Dropbox/output/asin_list_name.csv')
    mov_list = df_list['asin'].tolist()
    df = df[df['asin'].isin(mov_list)]

    s = df['asin'].value_counts().sort_values(ascending=False).head(n_movies)
    print(s)
    new_df = pd.DataFrame({'asin':s.index}).merge(df, how='left')

    print(len(new_df))

    return new_df

# DETECT WHICH LANGUAGE IS USED IN EACH REVIEW
# IF A REVIEW IS DETECTED TO BE WRITTEN IN ANY LANGUAGE OTHER THAN ENGLISH
# THEN DROP IT
def remove_non_english(df):
    print("REMOVING NON ENGLISH REVIEWS")

    # df[df.reviewText.apply(lambda x : detect(x))!='en']
    for i, row in df.iterrows():
        review = row['reviewText']
        lang = detect(str(review))
        if lang!='en':
            print(lang, review)
            df.drop(i, inplace=True)

    print(len(df))

    return df

# TOKENIZATION
def tokenize_df(df):
    df_token = []
    for review in df['reviewText']:
        temp = review
        temp = re.sub("[^a-zA-Z']", " ", str(review))
        temp = temp.replace("'", "")
        temp = temp.lower()
        word_length = len(word_tokenize(temp))

        df_token.append({'reviewText': temp, 'word':word_length})

    df_token = pd.DataFrame(df_token)

    return df_token

# REMOVE REVIEWS WITH TOO MANY OR TOO FEW WORDS
def remove_short_long(df, min_words, max_words):
    print("REMOVING TOO SHORT AND TOO LONG REVIEWS")

    new_df = tokenize_df(df)

    # GET STATISTICS OF WORD LENGTH IN REVIEWS
    print(new_df['word'].describe())

    too_long = df.loc[new_df['word'] >= max_words, 'reviewText']
    too_short = df.loc[new_df['word'] <= min_words, 'reviewText']
    print('too long:', len(too_long))
    print('too short:', len(too_short))

    df['word'] = new_df['word']

    del_id = too_long.index.append(too_short.index)
    new_df = df.drop(df.index[[del_id]])

    print(len(new_df))

    return new_df

# MAIN PROGRAM
def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/full/helpful.csv'
    input_file = '/home/lia/Documents/the_project/dataset/helpfulness/zero.csv'
    df = pd.read_csv(input_file)
    df.rename( columns={'Unnamed: 0':'review_id'}, inplace=True )
    print(df.head(10))

    temp = get_top_movies(df, 30)
    temp = remove_short_long(temp, 0, 1000)
    temp = remove_non_english(temp)

    temp.to_csv("/home/lia/Dropbox/output/additional_dataset/cleaned_data.csv")

if __name__ == "__main__":
    main()
