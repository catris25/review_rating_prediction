import preprocessing_step as prep

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

from collections import Counter

import pandas as pd
import numpy as np

def classify_report(X_train_vectorized, y_train, X_test_vectorized, y_test):
    # FIT INTO CLASSIFIER
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


# GENERATE A DICTIONARY THAT WILL DECIDE HOW MANY RESAMPLES FOR EACH CLASS
def ratio_dict(old_ratio):
    new_ratio = {}

    for i in range(1,len(old_ratio)):
        new_ratio[i] = old_ratio[i]*2
        if i==2:
            new_ratio[i] = old_ratio[i]*3

    return new_ratio


def classify_nb(train_df, test_df):
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
    X_test_vectorized = vectorizer.transform(X_test)

    # Train and test data with classifier
    print("** CLASSIFICATION **")
    print('Original data {}'.format (Counter(y_train)))
    print('Testing data {}'. format(Counter(y_test)))
    classify_report(X_train_vectorized, y_train, X_test_vectorized, y_test)


    # OVERSAMPLE WITH SMOTE
    sm = SMOTE(ratio=ratio_dict(Counter(y_train)))
    X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

    print("** SMOTE + CLASSIFICATION **")
    print('Original data {}'.format (Counter(y_train)))
    print('Resampled data {}'.format(Counter(y_res)))
    print('Testing data {}'. format(Counter(y_test)))

    # Train and test data with NB classifier
    X_train_vectorized, y_train = X_res, y_res

    classify_report(X_train_vectorized, y_train, X_test_vectorized, y_test)

    # FEATURE SELECTION
    k = int(X_train_vectorized.shape[1]/2)
    selector = SelectKBest(chi2, k)
    X_select = selector.fit_transform(X_train_vectorized, y_train)

    print("** SMOTE + FEATURE SELECTION + CLASSIFICATION **")
    print(X_train_vectorized.shape)
    print(X_select.shape)

    X_train_vectorized = X_select
    X_test_vectorized = selector.transform(X_test_vectorized)

    classify_report(X_train_vectorized, y_train, X_test_vectorized, y_test)


def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/2.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/top_10_movies/top_10_clean.csv'
    # df = pd.read_csv(input_file)
    #
    # print("executing preprocessing step")
    # prep_df = prep.preprocess_data(df)

    input_file = '/home/lia/Documents/the_project/dataset/output/temp.csv'
    prep_df = pd.read_csv(input_file)

    # SPLIT INTO TRAINING AND TESTING
    train_df, test_df = train_test_split(prep_df, test_size=0.33)


    # USE prep_df AS TRAINING DATA AGAINST THE TESTING DATA
    classify_nb(train_df, test_df)


if __name__ == "__main__":
    main()
