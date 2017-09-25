import pandas as pd
import numpy as np

from langdetect import detect

# input_file = '/home/lia/Documents/the_project/dataset/musical_inst/helpful.csv'
local_file ='/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/20percent/5.csv'
df = pd.read_csv(local_file)

for i, row in df.iterrows():
    review = row['reviewText']
    lang = detect(str(review))
    if lang!='en':
        print(lang)
        print(review)
        df.drop(i, inplace=True)

# new_df = df[detect(str(df['reviewText']))!='en']
# new_df = df[] [detect(str(df['reviewText']))!='en']

# print(df)
# df.to_csv((local_file+'helpful_en.csv'),encoding="utf-8", sep=",")
