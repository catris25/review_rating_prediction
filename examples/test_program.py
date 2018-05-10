from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# MULTINOMIAL NAIVE BAYES
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

    df_conf_matrix = pd.DataFrame(conf_matrix, columns=[1,2,3,4,5])
    df_conf_matrix.index = np.arange(1, len(df_conf_matrix) + 1)

    return(df_conf_matrix, y_pred_class)

# LOGISTIC REGRESSION
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

    df_conf_matrix = pd.DataFrame(conf_matrix, columns=[1,2,3,4,5])
    df_conf_matrix.index = np.arange(1, len(df_conf_matrix) + 1)

    return(df_conf_matrix, y_pred_class)

# VECTORIZE DATA INTO A SPARSE MATRIX
def vectorize_data(X_train, X_test):
    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return (X_train_vectorized, X_test_vectorized)

def classify_data(df, n_loop):

    # DECLARE LISTS TO SAVE y_pred_class
    nb1_y = []
    # review_id_list = test_df[['review_id']]
    nb_y_list = []
    lr_y_list = []

    for i in range(0, n_loop):
        print("ITERATION-%d"%i)
        # SPLIT INTO TRAINING AND TESTING
        train_df, test_df = train_test_split(df, test_size=0.3)

        # READ TRAINING DATA AND SEPARATE INTO X AND y
        X_train = train_df['reviewText']
        y_train = train_df['overall']

        # READ TESTING DATA AND SEPARATE INTO X AND y
        X_test = test_df['reviewText']
        y_test = test_df['overall']

        # VECTORIZE THE DATA
        X_train_vectorized, X_test_vectorized = vectorize_data(X_train, X_test)

        # CALL EACH FUNCTION AND SAVE TO CORRESPONDING VARIABLE
        print("Multinomial Naive Bayes")
        nb, nb_y = classify_nb_report(X_train_vectorized, y_train, X_test_vectorized, y_test)
        print("Logistic Regression")
        logreg, lr_y = classify_logreg_report(X_train_vectorized, y_train, X_test_vectorized, y_test)

        print("\nDATA RATIO")
        print('Training data \t{}'.format (Counter(y_train)))
        print('Testing data \t{}'. format(Counter(y_test)))

        # new_df.append(test_df[['review_id', 'overall']].set_index('review_id').T.to_dict('list'))
        # if i==0:
        #     new_list = zip(test_df['review_id'], nb_y)
        #     new_dict = {key: value for (key, value) in new_list}
        # else:
        #     for key, val in

        nb_y_list.extend([list(x) for x in zip(test_df['review_id'], nb_y)])
        lr_y_list.extend([list(x) for x in zip(test_df['review_id'], lr_y)])
        # for f, b in zip(test_df['review_id'], lr_y):
        #     print(f, b)

        # END OF LOOP

    # dd1 = defaultdict(list)
    # for key, val in nb_y_list:
    #     dd1[key].append(val)
    # print(dd1)

    dd2 = defaultdict(list)
    for key, val in lr_y_list:
        dd2[key].append(val)
    print(dd2)

    mydict = {k:float(sum(v))/len(v) for k, v in dd2.items()}
    print(mydict)


def main():
    input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_5.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/clean_data.csv'

    prep_df = pd.read_csv(input_file)

    # PRINT STATS OF DATA
    n_reviews = len(prep_df)
    n_movies = len(prep_df['asin'].value_counts())
    print(" %d reviews of %d movies"%(n_reviews, n_movies))
    print(prep_df['overall'].value_counts().sort_index())

    n_loop = 5
    classify_data(prep_df, n_loop)


if __name__ == "__main__":
    main()

# from collections import defaultdict
#
# d1 = defaultdict(list)
#
# l = [[1, 'A'], [1, 'B'], [2, 'C']]
#
# for k, v in l:
#     print(v)
#     d1[k].append(v)
#
# d = dict((k, tuple(v)) for k, v in d1.items())
# print(d)
