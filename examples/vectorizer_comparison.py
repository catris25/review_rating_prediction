from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

from imblearn.over_sampling import SMOTE

from collections import Counter

import pandas as pd
import numpy as np

np.set_printoptions(linewidth=200)
np.set_printoptions(precision=1)

input_file = '/home/lia/Documents/the_project/dataset/output/test_data_more.csv'
df = pd.read_csv(input_file)

print(df['overall'].value_counts())

train_df, test_df = train_test_split(df, test_size=0.3)

print("TRAINING SET")
print(train_df)

print("TESTING SET")
print(test_df)

# READ TRAINING DATA AND SEPARATE INTO X AND y
X_train = train_df['reviewText']
y_train = train_df['overall']

# READ TESTING DATA AND SEPARATE INTO X AND y
X_test = test_df['reviewText']
y_test = test_df['overall']

# VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
print(vectorizer.vocabulary_)

X_test_vectorized = vectorizer.transform(X_test)

print(X_train_vectorized.toarray())

sm = SMOTE(k_neighbors=1)
X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

print('Original data {}'.format (Counter(y_train)))
print('Resampled data {}'. format(Counter(y_res)))

print(X_res.toarray())

X_train_vectorized = X_res
y_train = y_res

# clf = MultinomialNB()
clf = LogisticRegression()

# TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
clf.fit(X_train_vectorized, y_train)


y_pred_class = clf.predict(X_test_vectorized)

accu = metrics.accuracy_score(y_test, y_pred_class)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

report_matrix = metrics.classification_report(y_test, y_pred_class)
print(report_matrix)
