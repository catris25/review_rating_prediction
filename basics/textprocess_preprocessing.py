import pandas as pd
import numpy as np

from textprocess import tokenizer

input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/20percent/4.csv'
orig_df = pd.read_csv(input_file)
# print(orig_df['overall'].value_counts().sort_index())
