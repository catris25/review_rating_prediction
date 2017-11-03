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


def sep_to_x_y(df):
    reviews = df['reviewText']
    X_data = [review.strip() for review in reviews]
    y_data = df['overall']

    return X_data, y_data

input_file = '/home/lia/Documents/the_project/dataset/output/clean_df.csv'
# input_file = '/home/lia/Documents/the_project/dataset/output/clean_data.csv'
# input_file = '/home/lia/Documents/the_project/dataset/musical_inst/clean_data.csv'
# input_file = "/home/lia/Documents/the_project/dataset/clean_airline_sentiments.csv"

orig_df = pd.read_csv(input_file)

# orig_df = orig_df.drop('summary', axis=1)

print(orig_df['overall'].value_counts().sort_index())
print(orig_df.head(5))

# CALCULATING ACCURACY
X_data = orig_df['reviewText']
y_data = orig_df['overall']

# vect = TfidfVectorizer(binary=True, min_df=5, ngram_range=(1,3))
vect = CountVectorizer(binary=True, min_df=5, ngram_range=(1,1))
# vect = HashingVectorizer()

X_dtm = vect.fit_transform(X_data.values.astype('U'))

print(X_dtm.toarray().shape)

mnb = MultinomialNB()
bnb = BernoulliNB()

# knn = KNeighborsClassifier(n_neighbors=5)
# svr = svm.SVC(kernel='linear',
#             class_weight='balanced', # penalize
#             probability=True)

# logreg = linear_model.LogisticRegression(penalty="l2", C=1)
# rf = RandomForestClassifier(random_state=123)
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,5), random_state=125)

# ensemble_voting = VotingClassifier(estimators=[('logreg', logreg),('mnb', mnb),('svr', svr), ('mlp', mlp)], voting='hard')
# bagging = BaggingClassifier(base_estimator=bnb, n_estimators=100, random_state=123)

clf = mnb

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.3,random_state = 41)

# FIT INTO CLASSIFIER
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)

accu = accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

report_matrix = metrics.classification_report(y_test, y_pred_class)
print(report_matrix)

# y_test = np.asarray(y_test)
# incorrect = np.where(y_test != y_pred_class)
# orig_df['prediction'] = y_pred_class.tolist()
#
# df_incorrect = orig_df.ix[incorrect][['reviewText', 'overall', 'prediction']]
# print(df_incorrect)

# class_labels = [0,1]
# feature_names = vect.get_feature_names()
# for i, class_label in enumerate(class_labels):
#     top10 = np.argsort(clf.coef_[i])[-15:]
#     print("%s: %s" % (class_label,
#           ". ".join(feature_names[j] for j in top10)))
