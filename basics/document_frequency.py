import pandas as pd
import numpy as np
import ast, sys

input_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'

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

doc_freq_df = pd.DataFrame()
doc_freq_df['word'] = word_list

for rtg in ratings:
    data_per_class = df.loc[(df['overall'] == rtg)]
    current_reviews = data_per_class['reviewText']
    d = []

    for word in word_list:
        freq = sum(1 for text in current_reviews if word in text)
        temp = freq/len(data_per_class)

        d.append(temp)

    max_df_class = max(d)

    fr = []
    for i in d:
        freq_rel = i/max_df_class
        fr.append(freq_rel)

    rtg = str(rtg)
    rtg = "class_"+rtg
    # doc_freq_df[rtg] = d
    doc_freq_df[rtg] = fr

# find the biggest values
doc_freq_df['max'] = doc_freq_df[['class_1','class_2','class_3','class_4','class_5']].idxmax(axis=1)
print(doc_freq_df.head(50))

# calculate the difference
# new_df = doc_freq_df[[1,2,3,4,5]]
new_df = doc_freq_df[['class_1','class_2','class_3','class_4','class_5']]

threshold_val = 0.075

passed_words = []
for row in doc_freq_df.itertuples():
    # print(row.word)
    val = [row.class_1, row.class_2, row.class_3, row.class_4, row.class_5]

    passed = False
    for i in range(5):
        for j in range(i,5):
            temp = abs(val[i]-val[j])
            if temp >= threshold_val:
                passed = True
                passed_words.append(row.word)

            if passed:
                break
        if passed:
            break

print(passed_words)
new_df = []

df_filtering = []
for review in df['reviewText']:
    filtered_text = [word for word in review if word in passed_words]
    df_filtering.append(filtered_text)

df['reviewText'] = df_filtering
# print(df['reviewText'])

output_file = '/home/lia/Documents/the_project/dataset/output/clean_df.csv'
df.to_csv(output_file, index=False, sep=',')

# NOTES
# calculate the difference between word in c1, c2, c3, so on
# calculate the df per max df in that class
