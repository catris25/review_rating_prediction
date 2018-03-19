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

    # incorrect = np.where(y_test != y_pred_class)
    # return incorrect


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

    # for i in range(1,len(old_ratio)):
    #     new_ratio[i] = old_ratio[i]*2
    #     if i==2:
    #         new_ratio[i] = old_ratio[i]*3

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

    # rus = RandomUnderSampler(ratio={5:1200})
    # X_res, y_res = rus.fit_sample(X_res, y_res)

    print("** SMOTE + CLASSIFICATION **")
    print('Original data {}'.format (Counter(y_train)))
    print('Resampled data {}'.format(Counter(y_res)))
    print('Testing data {}'. format(Counter(y_test)))

    # Train and test data with NB classifier
    X_train_vectorized, y_train = X_res, y_res

    classify_report(X_train_vectorized, y_train, X_test_vectorized, y_test)

    # FEATURE SELECTION
    # k = int(X_train_vectorized.shape[1]/2)
    # # k = int(X_train_vectorized.shape[0]/10)
    # selector = SelectKBest(chi2, k)
    # X_select = selector.fit_transform(X_train_vectorized, y_train)
    #
    # print("** SMOTE + FEATURE SELECTION + CLASSIFICATION **")
    # print(X_train_vectorized.shape)
    # print(X_select.shape)
    #
    # X_train_vectorized = X_select
    # X_test_vectorized = selector.transform(X_test_vectorized)
    #
    # incorrect = classify_report(X_train_vectorized, y_train, X_test_vectorized, y_test)
    # return incorrect


def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/8.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/top_50_movies/helpful.csv'
    input_file = "/home/lia/Documents/the_project/dataset/to_use/clean.csv"
    df = pd.read_csv(input_file)

    print("executing preprocessing step")
    prep_df = prep.preprocess_data(df)

    input_file = '/home/lia/Documents/the_project/dataset/output/temp.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/output/temp_top_50.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/output/clean_large_data.csv'
    prep_df = pd.read_csv(input_file)

    # SPLIT INTO TRAINING AND TESTING
    train_df, test_df = train_test_split(prep_df, test_size=0.3)

    # PRINT STATS OF DATA
    n_reviews = len(prep_df)
    n_movies = len(prep_df['asin'].value_counts())
    print(" %d reviews of %d movies"%(n_reviews, n_movies))
    print(prep_df['overall'].value_counts().sort_index())

    # PASS TO CLASSIFIER AND GATHER INFO OF THE MISCLASSIFIED DATA
    incorrect = classify_nb(train_df, test_df)

    # print("\nMISCLASSIFIED")
    # incorrect_asin = prep_df.ix[incorrect]['asin'].value_counts()
    # print(incorrect_asin)


if __name__ == "__main__":
    main()
