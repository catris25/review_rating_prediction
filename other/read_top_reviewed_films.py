import pandas as pd


input_file = '/home/lia/Documents/the_project/dataset/helpfulness/zero.csv'

df = pd.read_csv(input_file)
reviews_freq = df['asin'].value_counts().sort_values(ascending=False)
s = reviews_freq[reviews_freq>=100]

print(s)
print(s.sum())
