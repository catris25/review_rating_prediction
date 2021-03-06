import pandas as pd
import numpy as np
import re, string
import sys
# import textmining

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, grid_search
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.cross_validation import train_test_split


input_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'
# input_file = '/home/lia/Documents/the_project/dataset/musical_inst/clean_data.csv'
# input_file = "/home/lia/Documents/the_project/dataset/clean_airline_sentiments.csv"

orig_df = pd.read_csv(input_file)

# orig_df['orig_overall'] = orig_df['overall']


# orig_df.loc[orig_df['overall'] == 1 , 'overall'] = 0
# orig_df.loc[orig_df['overall'] == 5 , 'overall'] = 0

orig_df.loc[orig_df['overall'] == 2 , 'overall'] = 0
orig_df.loc[orig_df['overall'] == 3 , 'overall'] = 0
orig_df.loc[orig_df['overall'] == 4 , 'overall'] = 0

orig_df = orig_df[orig_df['overall'] != 0]

# orig_df.loc[orig_df['overall'] == 1, 'overall'] = 10
# orig_df.loc[orig_df['overall'] == 2, 'overall'] = 0
# orig_df.loc[orig_df['overall'] == 3, 'overall'] = 0
# orig_df.loc[orig_df['overall'] == 4, 'overall'] = 0
# orig_df.loc[orig_df['overall'] == 5, 'overall'] = 10




print(orig_df['overall'].value_counts().sort_index())
print(orig_df.head(5))

# CALCULATING ACCURACY
# X_data, y_data = sep_to_x_y(orig_df)
X_data = orig_df['reviewText']
y_data = orig_df['overall']

# vect = TfidfVectorizer(binary=True, min_df=5, ngram_range=(1,3))
vect = CountVectorizer(binary=True, min_df=3)
# vect = HashingVectorizer()

X_dtm = vect.fit_transform(X_data.values.astype('U'))

# print(vect.get_feature_names())

print(X_dtm.toarray().shape)

mnb = MultinomialNB()
bnb = BernoulliNB()

clf = mnb

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.33,random_state = 33)

# FIT INTO CLASSIFIER
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)

accu = accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

report_matrix = metrics.classification_report(y_test, y_pred_class)
print(report_matrix)
