import pandas as pd
import numpy as np
import re, string, ast
import sys
import nltk, numbers
from collections import Counter

input_file = '/home/lia/Documents/the_project/dataset/output/test_data.csv'

df = pd.read_csv(input_file)
print(df['overall'].value_counts().sort_index())
print(df.head(5))

ratings = [1,2,3,4,5]

# read data from column as list in list
reviews = []
for r in df['reviewText']:
    r = ast.literal_eval(r)
    reviews.append(r)

df['reviewText'] = reviews

# get list of set of all words in the corpus
all_words = []
for review in reviews:
    for word in review:
        all_words.append(word)

word_list = list(set(all_words))
word_list.sort()

# create a dict to count the df of all words in the corpus
total_df = {}
for word in word_list:
    doc_freq = sum(1 for text in reviews if word in text)
    total_df[word] = doc_freq

# create a dict to count the document frequency in each class
# class_df = {}
# for rtg in ratings:
#     class_df[rtg] =

def calc_docfreq(dpc):
    current_df = {}
    for word in word_list:
        doc_freq = sum(1 for text in dpc if word in text)
        current_df[word] = doc_freq
    return current_df

def calc_concentration_degree(cdf):
    cd = {}
    for word in word_list:
        cd[word] = cdf[word]/(total_df[word]-cdf[word]+1)
    return cd

concentration_dict = {}
for rtg in ratings:

    # get doc freq (df) of each word in each class
    print("\nrating %d stars"%rtg )
    data_per_class = df.loc[(df['overall'] == rtg)]
    current_df = calc_docfreq(data_per_class['reviewText'])

    # calculate the concentration degree
    concent_deg = calc_concentration_degree(current_df)

    sorted_dict = sorted(((value,key) for (key,value) in concent_deg.items()), reverse=True)
    print(sorted_dict[:10])

    concentration_dict[rtg] = concent_deg

# DISPERSE DEGREE
disperse_dict = {}
for rtg in ratings:
    docs_len = len(df[df['overall'] == rtg])

    data_per_class = df.loc[(df['overall'] == rtg)]
    current_df = calc_docfreq(data_per_class['reviewText'])

    disperse_deg = {}
    for word in word_list:
        disperse_deg[word] = current_df[word]/docs_len

    # print(disperse_deg)
    disperse_dict[rtg] = disperse_deg

# CONTRIBUTION DEGREE
