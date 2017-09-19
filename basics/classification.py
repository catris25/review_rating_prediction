import pandas as pd
import numpy as np
import re, string
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, grid_search
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.cross_validation import train_test_split


def sep_to_x_y(df):
    reviews = df['reviewText']
    X_data = [review.strip() for review in reviews]
    y_data = df['overall']

    return X_data, y_data

input_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'
# input_file = "/home/lia/Documents/the_project/dataset/clean_airline_sentiments.csv"

orig_df = pd.read_csv(input_file)

print(orig_df['overall'].value_counts().sort_index())
print(orig_df.head(5))

# CALCULATING ACCURACY
X_data, y_data = sep_to_x_y(orig_df)

# vect = TfidfVectorizer(binary=True, min_df=5, ngram_range=(1,2))
vect = CountVectorizer(binary=True, min_df=3, ngram_range=(1,3))
# vect = HashingVectorizer()

X_dtm = vect.fit_transform(X_data)

# print(vect.get_feature_names())

print(X_dtm.toarray().shape)

mnb = MultinomialNB()
knn = KNeighborsClassifier(n_neighbors=5)
svr = svm.SVC(kernel='rbf',C=12.0,gamma=0.001)
logreg = linear_model.LogisticRegression()
rf = RandomForestClassifier(random_state=123)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,2,5), random_state=12)
# best mlp [2,4] [3,4,6]

ensemble_voting = VotingClassifier(estimators=[('logreg', logreg),('mnb', mnb)], voting='soft')
bagging = BaggingClassifier(base_estimator=knn, n_estimators=100, random_state=123)

clf = knn

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.33, random_state=456)

# FIT INTO CLASSIFIER
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)

accu = accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

# class_labels = [1,2,3,4,5]
# feature_names = vect.get_feature_names()
# for i, class_label in enumerate(class_labels):
#     top10 = np.argsort(clf.coef_[i])[-15:]
#     print("%s: %s" % (class_label,
#           ". ".join(feature_names[j] for j in top10)))
