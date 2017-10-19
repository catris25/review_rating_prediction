import pandas as pd
import numpy as np
import re, string, ast, math
import sys
import nltk, numbers
from collections import Counter

input_file = '/home/lia/Documents/the_project/dataset/output/test_data.csv'
# input_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'

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

print("CONCENTRATION DEGREE")
concentration_dict = {}
for rtg in ratings:
    print("\nrating %d stars"%rtg )

    # get doc freq (df) of each word in each class
    data_per_class = df.loc[(df['overall'] == rtg)]
    current_df = calc_docfreq(data_per_class['reviewText'])

    # calculate the concentration degree
    concent_deg = calc_concentration_degree(current_df)

    sorted_dict = sorted(((value,key) for (key,value) in concent_deg.items()), reverse=True)
    # print(sorted_dict[:10])
    print(concent_deg)

    concentration_dict[rtg] = concent_deg

# DISPERSE DEGREE
print("\nDISPERSE DEGREE")
disperse_dict = {}
for rtg in ratings:
    print("\nrating %d stars"%rtg )
    docs_len = len(df[df['overall'] == rtg])

    data_per_class = df.loc[(df['overall'] == rtg)]
    current_df = calc_docfreq(data_per_class['reviewText'])

    disperse_deg = {}
    for word in word_list:
        disperse_deg[word] = current_df[word]/docs_len

    print(disperse_deg)
    disperse_dict[rtg] = disperse_deg

# CONTRIBUTION DEGREE
print("\nCONTRIBUTION DEGREE")

contribution_dict = {}
for rtg in ratings:
    print("\nrating %d stars"%rtg )

    # get doc freq (df) of each word in each class
    data_in_c = df.loc[(df['overall'] == rtg)]
    corpus_len = len(df)
    contrib_deg = {}

    for word in word_list:
        doc_freq = sum(1 for text in data_in_c['reviewText'] if word in text)
        if(doc_freq>0):
            # p_ct = doc_freq/corpus_len
            # p_cnt = doc_freq/total_df[word]
            p_cnt = doc_freq/len(data_in_c)
            p_c = len(data_in_c)/corpus_len

            temp = p_cnt/p_c
            contrib_deg[word] = p_cnt*math.log2(temp)
        else:
            p_ct = 0
            p_cnt = 0
            p_c = 0
            contrib_deg[word] = 0

        # print(doc_freq)
        # print(p_ct)
        # print(p_cnt)
        # print(p_c)

    print("cd ",contrib_deg)
    contribution_dict[rtg] = contrib_deg
