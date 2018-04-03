import numpy as np
import pandas as pd
import re, math
from collections import Counter

# input_file='/home/lia/Documents/the_project/dataset/to_use/top_50.csv'
input_file='/home/lia/Documents/the_project/dataset/to_use/current/top_10.csv'
# input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/6.csv'

df = pd.read_csv(input_file)

print(df['asin'].value_counts())

# df['asin'].value_counts().to_csv('/home/lia/Documents/the_project/dataset/to_use/current/asin_list_name.csv')
