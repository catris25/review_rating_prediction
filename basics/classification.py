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

input_file = '/home/lia/Documents/the_project/dataset/output/rep_data.csv'
# input_file = '/home/lia/Documents/the_project/dataset/musical_inst/clean_data.csv'
# input_file = "/home/lia/Documents/the_project/dataset/clean_airline_sentiments.csv"

orig_df = pd.read_csv(input_file)

print(orig_df['overall'].value_counts().sort_index())
print(orig_df.head(5))

# CALCULATING ACCURACY
# X_data, y_data = sep_to_x_y(orig_df)
X_data = orig_df['reviewText']
y_data = orig_df['overall']

# vect = TfidfVectorizer(binary=True, min_df=3, max_df=0.3, ngram_range=(1,3))
vect = CountVectorizer(binary=True, min_df=5, ngram_range=(1,3))
# vect = HashingVectorizer()

X_dtm = vect.fit_transform(X_data.values.astype('U'))

# print(vect.get_feature_names())

print(X_dtm.toarray().shape)

mnb = MultinomialNB()
bnb = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
knn = KNeighborsClassifier(n_neighbors=5)
# svr = svm.SVC(kernel='rbf',C=12.0,gamma=0.001)
svr = svm.SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

logreg = linear_model.LogisticRegression(penalty="l2", C=1)
rf = RandomForestClassifier(random_state=123)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,4,6), random_state=12)
# best mlp [2,4] [3,4,6]

ensemble_voting = VotingClassifier(estimators=[('logreg', logreg),('mnb', mnb),('svr', svr), ('mlp', mlp)], voting='hard')
bagging = BaggingClassifier(base_estimator=bnb, n_estimators=100, random_state=123)

clf = bnb

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.25, random_state=123)

# FIT INTO CLASSIFIER
clf.fit(X_train, y_train)

y_pred_class = clf.predict(X_test)

accu = accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

report_matrix = metrics.classification_report(y_test, y_pred_class)
print(report_matrix)

# class_labels = [0,1]
# feature_names = vect.get_feature_names()
# for i, class_label in enumerate(class_labels):
#     top10 = np.argsort(clf.coef_[i])[-15:]
#     print("%s: %s" % (class_label,
#           ". ".join(feature_names[j] for j in top10)))
