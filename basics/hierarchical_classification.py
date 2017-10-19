import pandas as pd
import numpy as np
import re, string
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.cross_validation import train_test_split

input_file = '/home/lia/Documents/the_project/dataset/output/rep_data.csv'

orig_df = pd.read_csv(input_file)

print(orig_df['overall'].value_counts().sort_index())
print(orig_df.head(5))

# CALCULATING ACCURACY
X_data = orig_df['reviewText']
y_data = orig_df['overall']

# vect = TfidfVectorizer(binary=True, min_df=5, ngram_range=(1,3))
vect = CountVectorizer(binary=True, min_df=5, ngram_range=(1,3))

X_dtm = vect.fit_transform(X_data.values.astype('U'))

print(X_dtm.toarray().shape)

mnb = MultinomialNB()
bnb = BernoulliNB()

# knn = KNeighborsClassifier(n_neighbors=5)

clf = mnb

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.5,random_state = 44)

# FIT INTO CLASSIFIER
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)

accu = accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

# report_matrix = metrics.classification_report(y_test, y_pred_class)
# print(report_matrix)

incorrectly_zero = np.nonzero((y_pred_class != y_test) & (y_pred_class==0))
correctly_zero = np.nonzero((y_pred_class == y_test) & (y_pred_class==0))
df_temp = orig_df.ix[correctly_zero].append(orig_df.ix[incorrectly_zero])
# df_temp['overall'] = np.where(df_temp['overall']<i, '0', i)

# classify 4 and not 4 (1,2,3) and so on
for i in range(4,1,-1):
    df_temp['overall'] = np.where(df_temp['orig_overall']==i, i, 0)

    X_data = df_temp['reviewText']
    y_data = df_temp['overall']
    X_dtm = vect.fit_transform(X_data.values.astype('U'))

    print(df_temp['overall'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.5,random_state = 23)
    print(X_dtm.toarray().shape)

    clf.fit(X_train, y_train)

    y_pred_class = clf.predict(X_test)

    accu = accuracy_score(y_pred_class, y_test)
    print(accu)

    conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
    print(conf_matrix)

    incorrectly_zero = np.nonzero((y_pred_class != y_test) & (y_pred_class==0))
    correctly_zero = np.nonzero((y_pred_class == y_test) & (y_pred_class==0))
    df_temp = orig_df.ix[correctly_zero].append(orig_df.ix[incorrectly_zero])

    # print(df_temp.head(10))
