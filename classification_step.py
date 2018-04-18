import preprocessing_step as prep

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from collections import Counter

import pandas as pd
import numpy as np

# GENERATE A DICTIONARY THAT WILL DECIDE HOW MANY RESAMPLES FOR EACH CLASS
def ratio_dict(old_ratio):
    new_ratio = {}
    max_r = max(old_ratio)

    # LOOP THROUGH DICTIONARY ORDERED BY HIGHEST VALUE
    for key in sorted(old_ratio, key=old_ratio.get, reverse=True):
        if key == max_r:
            curr_max = old_ratio[key]
            new_ratio[key] = old_ratio[key]
            continue
        else:
            diff = int(curr_max/old_ratio[key])
            new_ratio[key] = old_ratio[key] * diff
            curr_max = new_ratio[key]
            print("%d. %d x %d = %d"%(key, diff, old_ratio[key], new_ratio[key]))

    return new_ratio

def oversample_proportional(X_train_vectorized, y_train):
    # OVERSAMPLE WITH PROPORTIONAL RATIO SMOTE
    sm = SMOTE(ratio=ratio_dict(Counter(y_train)))
    X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

    return (X_res, y_res)

def oversample_unproportional(X_train_vectorized, y_train):
    # OVERSAMPLE WITH UNPROPORTIONAL SMOTE
    sm = SMOTE(ratio='all')
    X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

    return (X_res, y_res)

def classify_knn_report(X_train_vectorized, y_train, X_test_vectorized, y_test):
    # FIT INTO CLASSIFIER
    clf = KNeighborsClassifier(n_neighbors=5)

    # TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
    clf.fit(X_train_vectorized, y_train)

    y_pred_class = clf.predict(X_test_vectorized)

    accu = metrics.accuracy_score(y_test, y_pred_class)
    print(accu)

    conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
    print(conf_matrix)

    report_matrix = metrics.classification_report(y_test, y_pred_class)
    # print(report_matrix)


def classify_nb_report(X_train_vectorized, y_train, X_test_vectorized, y_test):
    # FIT INTO CLASSIFIER
    clf = MultinomialNB()

    # TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
    clf.fit(X_train_vectorized, y_train)

    y_pred_class = clf.predict(X_test_vectorized)

    accu = metrics.accuracy_score(y_test, y_pred_class)
    print(accu)

    conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
    print(conf_matrix)

    report_matrix = metrics.classification_report(y_test, y_pred_class)
    # print(report_matrix)

def classify_logreg_report(X_train_vectorized, y_train, X_test_vectorized, y_test):
    # FIT INTO CLASSIFIER
    clf = LogisticRegression()

    # TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
    clf.fit(X_train_vectorized, y_train)

    y_pred_class = clf.predict(X_test_vectorized)

    accu = metrics.accuracy_score(y_test, y_pred_class)
    print(accu)

    conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
    print(conf_matrix)
    
    report_matrix = metrics.classification_report(y_test, y_pred_class)

def vectorize_data(X_train, X_test):
    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return (X_train_vectorized, X_test_vectorized)

def classify_data(df):
    # SPLIT INTO TRAINING AND TESTING
    train_df, test_df = train_test_split(df, test_size=0.3)

    # READ TRAINING DATA AND SEPARATE INTO X AND y
    X_train = train_df['reviewText']
    y_train = train_df['overall']

    # READ TESTING DATA AND SEPARATE INTO X AND y
    X_test = test_df['reviewText']
    y_test = test_df['overall']

    X_train_vectorized, X_test_vectorized = vectorize_data(X_train, X_test)

    print("Multinomial Naive Bayes")
    classify_nb_report(X_train_vectorized, y_train, X_test_vectorized, y_test)
    print("Logistic Regression")
    classify_logreg_report(X_train_vectorized, y_train, X_test_vectorized, y_test)

    X_res_unp, y_res_unp = oversample_unproportional(X_train_vectorized, y_train)
    print("Unproportional SMOTE + MNB")
    classify_nb_report(X_res_unp, y_res_unp, X_test_vectorized, y_test)
    print("Unproportional SMOTE + LogReg")
    classify_logreg_report(X_res_unp, y_res_unp, X_test_vectorized, y_test)

    X_res_p, y_res_p = oversample_proportional(X_train_vectorized, y_train)
    print("Proportional SMOTE + MNB")
    classify_nb_report(X_res_p, y_res_p, X_test_vectorized, y_test)
    print("Proportional SMOTE + LogReg")
    classify_logreg_report(X_res_p, y_res_p, X_test_vectorized, y_test)


    print("\nDATA RATIO")
    print('Training data \t{}'.format (Counter(y_train)))
    print('Testing data \t{}'. format(Counter(y_test)))
    print('Resampled training data')
    print('Unproportional \t{}'.format(Counter(y_res_unp)))
    print('Proportional \t{}'.format(Counter(y_res_p)))



def main():
    input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/output/without_five.csv'

    prep_df = pd.read_csv(input_file)

    # SPLIT INTO TRAINING AND TESTING
    train_df, test_df = train_test_split(prep_df, test_size=0.3)

    # PRINT STATS OF DATA
    n_reviews = len(prep_df)
    n_movies = len(prep_df['asin'].value_counts())
    print(" %d reviews of %d movies"%(n_reviews, n_movies))
    print(prep_df['overall'].value_counts().sort_index())

    # CLASSIFY THE DATA AND REPEAT IT 30 TIMES
    for i in range(0,3):
        print("ITERATION-%d"%i)
        classify_data(prep_df)



if __name__ == "__main__":
    main()
