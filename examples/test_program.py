from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

import scipy.stats as st

from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# CALCULATE t SCORE
def calculate_t_score(pop_list, samp_list):
    n1 = len(samp_list)
    n2 = len(pop_list)

    mean1 = np.mean(samp_list)
    mean2 = np.mean(pop_list)

    std1 = np.std(samp_list, ddof=1)
    std2 = np.std(pop_list, ddof=1)
    var1 = std1**2
    var2 = std2**2

    standard_error = np.sqrt(((n1-1)*var1+(n2-1)*var2)/(n1+n2-2))
    t_score = (mean1 - mean2)/(standard_error*(np.sqrt((1/n1)+(1/n2))))
    prob = st.t.sf(np.abs(t_score), df=4)

    print("t :",t_score)
    print("p :",prob)

    alpha = 0.05
    if prob < alpha:
        diff_significance = True
    else:
        diff_significance = False

    return diff_significance

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

def calculate_review_scores(current_list):
    dd = defaultdict(list)
    for key, val in current_list:
        dd[key].append(val)

    avg_dict = {k:float(sum(v))/len(v) for k, v in dd.items()}
    print(avg_dict)
    return avg_dict

def calculate_film_scores(review_scores_dict, df):
    df = df.assign(prediction = df['review_id'].map(review_scores_dict))

    print(len(df))
    new_df = df.dropna(how='any')
    print(len(new_df))
    print(new_df.head(10))

    avg_prediction = new_df['prediction'].groupby(new_df['asin']).mean().reset_index()
    df_avg_prediction = pd.DataFrame(avg_prediction)
    print(df_avg_prediction)

    avg_actual = new_df['overall'].groupby(new_df['asin']).mean().reset_index()
    df_avg_actual = pd.DataFrame(avg_actual)
    print(df_avg_actual)

    df_comparison = pd.merge(df_avg_prediction, df_avg_actual, on="asin")
    print(df_comparison)

    return df_comparison

def classify_data(df, n_loop):

    # DECLARE LISTS TO SAVE y_pred_class
    nb_y_list = []
    lr_y_list = []

    # DECLATE LISTS TO SAVE t_scores FOR EACH TESTING
    significance_list = []

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

        print(lr_y)
        nb_y_list.extend([list(x) for x in zip(test_df['review_id'], nb_y)])
        lr_y_list.extend([list(x) for x in zip(test_df['review_id'], lr_y)])

        significance = calculate_t_score(lr_y, y_test)
        significance_list.append(significance)

        # END OF LOOP

    print(significance_list)
    lr_score = calculate_review_scores(lr_y_list)
    film_scores = calculate_film_scores(lr_score, df[['review_id', 'asin', 'overall']])


def main():
    input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_5.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/output/test_data.csv'

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
