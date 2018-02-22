import preprocessing_step as prep

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics

import pandas as pd

def classify_nb(train_df, test_df):
    # # CHECK TO SEE IF TRAINING DATA IS STILL IN DF FORMAT AND HAVEN'T BEEN VECTORIZED
    # # IF TRUE THEN VECTORIZE THEM
    # if isinstance(train_data, pd.DataFrame):
    #     X_train = train_data['reviewText']
    #     y_train = train_data['overall']
    #
    #     # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    #     vectorizer = CountVectorizer()
    #     X_train_vectorized = vectorizer.fit_transform(X_train)
    # READ TRAINING DATA AND SEPARATE INTO X AND y
    X_train = train_df['reviewText']
    y_train = train_df['overall']


    # READ TESTING DATA AND SEPARATE INTO X AND y
    X_test = test_df['reviewText']
    y_test = test_df['overall']


    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # OVERSAMPLE WITH SMOTE
    # sm = SMOTE(ratio='all')
    # X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)
    # X_train, y_train = X_res, y_res

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
    print(report_matrix)


def resample():
    # DO THE SMOTE THING

    # READ TRAINING DATA AND SEPARATE INTO X AND y
    X_train = train_df['reviewText']
    y_train = train_df['overall']

    # READ TESTING DATA AND SEPARATE INTO X AND y
    X_test = test_df['reviewText']
    y_test = test_df['overall']

    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = CountVectorizer()
    # vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # OVERSAMPLE WITH SMOTE
    sm = SMOTE(ratio='all')
    X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)


def main():
    input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/10percent/3.csv'
    df = pd.read_csv(input_file)

    # READ TESTING DATA
    input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/10percent/1.csv'
    test_df = pd.read_csv(input_file)

    print("executing preprocessing step")
    prep_df = prep.preprocess_data(df)

    # USE prep_df AS TRAINING DATA AGAINST THE TESTING DATA
    classify_nb(prep_df, test_df)


if __name__ == "__main__":
    main()
