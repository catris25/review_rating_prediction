import pandas as pd
from sklearn import datasets
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import sys

input_file = '/home/lia/Documents/the_project/dataset/output/clean_df.csv'
df = pd.read_csv(input_file)

print(df.head(15))
