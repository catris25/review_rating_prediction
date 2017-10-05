import pandas as pd
import numpy as np

input_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'

df = pd.read_csv(input_file)

# df.loc[df['overall'] < 3 , 'overall'] = -10
# df.loc[df['overall'] == 3 , 'overall'] = 0
# df.loc[df['overall'] > 3 , 'overall'] = 10

df.loc[df['overall'] < 5 , 'overall'] = 0
df.loc[df['overall'] == 5 , 'overall'] = 5

print(df.head(10))
print(df['overall'].value_counts())

output_file = '/home/lia/Documents/the_project/dataset/output/rep_data.csv'
df.to_csv(output_file, index=False, sep=',')
