# 1. get 20 or 30 top film reviews
# 2. separate on helpfulness
# 3. remove non-english reviews
# 4. remove irrelevant reviews (dvd not working, bluray, extended ver, etc)
# 5. remove too short reviews

# notes
# what to viusalize in text classification in a scatter plot?

import pandas as pd
import numpy as np

import re
from nltk.tokenize import word_tokenize
from langdetect import detect

def get_helpful():
    print("GETTING HELPFUL REVIEWS ONLY")
    return 0

def get_top_movies(df, n_movies):
    print("GETTING %d MOST REVIEWED MOVIES"%n_movies)

    # df_list = pd.read_csv('/home/lia/Documents/the_project/dataset/to_use/current/asin_list_avg.csv')
    # mov_list = df_list['asin'].tolist()
    # df = df[df['asin'].isin(mov_list)]
    # print(df['asin'].value_counts())

    s = df['asin'].value_counts().sort_values(ascending=False).head(n_movies)
    print(s)
    new_df = pd.DataFrame({'asin':s.index}).merge(df, how='left')

    print(len(new_df))

    return new_df

def get_random_reviews(df, n_reviews):
    print("GETTING %d RANDOM REVIEWS"%n_reviews)
    new_df = df.sample(n_reviews)
    #
    # print(new_df)
    # import sys
    # sys.exit("ok")

    return new_df

def remove_non_english(df):
    print("REMOVING NON ENGLISH REVIEWS")

    df[df.reviewText.apply(lambda x : detect(x))!='en']
    # for i, row in df.iterrows():
    #     review = row['reviewText']
    #     lang = detect(str(review))
    #     if lang!='en':
    #         df.drop(i, inplace=True)

    print(len(df))

    return df

def remove_irrelevant():
    return 0

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

def remove_short_long(df, min_words, max_words):
    print("REMOVING TOO SHORT AND TOO LONG REVIEWS")

    new_df = tokenize_df(df)

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
    input_file = '/home/lia/Documents/the_project/dataset/to_use/top_50.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/full/helpful.csv'
    df = pd.read_csv(input_file)
    print(len(df))

    # temp = get_top_movies(df, 30)
    temp = get_random_reviews(df, 5000)
    # temp = remove_short_long(temp, 10, 1000)
    # temp = remove_non_english(temp)

    temp.to_csv("/home/lia/Documents/the_project/dataset/to_use/current/random_5000.csv")

if __name__ == "__main__":
    main()
