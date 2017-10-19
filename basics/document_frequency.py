import pandas as pd
import numpy as np
import ast, sys

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

# all_df = pd.DataFrame(word_list, columns=('word'))
doc_freq_df = pd.DataFrame()
doc_freq_df['word'] = word_list
for rtg in ratings:
    # print("\nrating %d stars"%rtg )
    data_per_class = df.loc[(df['overall'] == rtg)]

    doc_freq = {}
    current_reviews = data_per_class['reviewText']

    d = []
    for word in word_list:
        freq = sum(1 for text in current_reviews if word in text)
        temp = freq/len(data_per_class)
        # doc_freq[word] = temp

        d.append(temp)

    doc_freq_df[rtg] = d

    # all_df[rtg] = doc_freq
    # print(doc_freq)




# find the biggest values
doc_freq_df['max'] = doc_freq_df[[1,2,3,4,5]].idxmax(axis=1)
print(doc_freq_df)
