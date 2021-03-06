from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# CALCULATE SCORE PER REVIEW
def calculate_review_scores(current_list):
    dd = defaultdict(list)
    for key, val in current_list:
        dd[key].append(val)

    avg_dict = {k:float(sum(v))/len(v) for k, v in dd.items()}
    print(avg_dict)
    return avg_dict

# CALCULATE SCORE PER FILM
def calculate_film_scores(review_scores_dict, df):
    # MAP THE REVIEW SCORES TO ITS CORRESPONDING review_ids
    df = df.assign(prediction = df['review_id'].map(review_scores_dict))
    new_df = df.dropna(how='any')

    avg_prediction = new_df['prediction'].groupby(new_df['asin']).mean().reset_index()
    df_avg_prediction = pd.DataFrame(avg_prediction)

    avg_actual = df['overall'].groupby(df['asin']).mean().reset_index()
    df_avg_actual = pd.DataFrame(avg_actual)

    df_comparison = pd.merge(df_avg_prediction, df_avg_actual, on="asin")
    print(df_comparison)

    return df_comparison

# SUM ALL MATRICES
def sum_all_matrices(matrix_list):
    sum_matrix = np.sum([df.iloc[:,0:5].values for df in matrix_list], axis=0)
    print(sum_matrix)
    avg_accu = sum(np.diag(sum_matrix))/sum_matrix.sum()
    print(avg_accu)

    matrix_df = pd.DataFrame(sum_matrix, columns=[1,2,3,4,5])
    # matrix_df.index = np.arange(1, len(matrix_df) + 1)
    # matrix_df['iteration'] = 100
    return matrix_df

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

# RESAMPLE THE SMALLER CLASSES TO THE PROPORTIONAL RATIO
def oversample_proportional(X_train_vectorized, y_train):
    # OVERSAMPLE WITH PROPORTIONAL RATIO SMOTE
    sm = SMOTE(ratio=ratio_dict(Counter(y_train)))
    X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

    return (X_res, y_res)

# RESAMPLE THE SMALLER CLASSES TO THE SAME NUMBER OF DATA IN MAJORITY CLASS
def oversample_unproportional(X_train_vectorized, y_train):
    # OVERSAMPLE WITH UNPROPORTIONAL SMOTE
    sm = SMOTE(ratio='all')
    X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

    return (X_res, y_res)

# def write_classification_report(report):
#     report_data = []
#     lines = report.split('\n')
#     for line in lines[2:-3]:
#         row = {}
#         row_data = line.split('      ')
#         row['class'] = row_data[0]
#         row['precision'] = float(row_data[1])
#         row['recall'] = float(row_data[2])
#         row['f1_score'] = float(row_data[3])
#         row['support'] = float(row_data[4])
#         report_data.append(row)
#     df_report = pd.DataFrame.from_dict(report_data)
#     return df_report

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

    # report_matrix = metrics.classification_report(y_test, y_pred_class)
    # print(report_matrix)

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

    # report_matrix = metrics.classification_report(y_test, y_pred_class)
    # print(report_matrix)

    return(df_conf_matrix, y_pred_class)

# VECTORIZE DATA INTO A SPARSE MATRIX
def vectorize_data(X_train, X_test):
    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return (X_train_vectorized, X_test_vectorized)

def classify_data(df, n_loop):

    # DECLARE LISTS TO SAVE THE DATAFRAMES CONTAINING CONFUSION MATRICES
    nb1_list = []
    logreg1_list = []

    nb2_list = []
    logreg2_list = []

    nb3_list = []
    logreg3_list = []

    # DECLARE LISTS TO SAVE y_pred_class
    nb1_y_list = []
    logreg1_y_list = []

    nb2_y_list = []
    logreg2_y_list = []

    nb3_y_list = []
    logreg3_y_list = []

    # LOOPING
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
        logreg, logreg_y = classify_logreg_report(X_train_vectorized, y_train, X_test_vectorized, y_test)

        X_res_unp, y_res_unp = oversample_unproportional(X_train_vectorized, y_train)
        print("Unproportional SMOTE + MNB")
        unp_smote_nb, unp_nb_y = classify_nb_report(X_res_unp, y_res_unp, X_test_vectorized, y_test)
        print("Unproportional SMOTE + LogReg")
        unp_smote_logreg, unp_logreg_y = classify_logreg_report(X_res_unp, y_res_unp, X_test_vectorized, y_test)

        X_res_p, y_res_p = oversample_proportional(X_train_vectorized, y_train)
        print("Proportional SMOTE + MNB")
        p_smote_nb, p_nb_y = classify_nb_report(X_res_p, y_res_p, X_test_vectorized, y_test)
        print("Proportional SMOTE + LogReg")
        p_smote_logreg, p_logreg_y = classify_logreg_report(X_res_p, y_res_p, X_test_vectorized, y_test)

        # ADD NEW COLUMN FOR EACH ITERATION AND APPEND TO THE LIST OF DATAFRAMES TO CORRESPONDING METHOD
        # TO SAVE CONFUSION MATRICS
        nb['iteration'] = i
        nb1_list.append(nb)
        unp_smote_nb['iteration'] = i
        nb2_list.append(unp_smote_nb)
        p_smote_nb['iteration'] = i
        nb3_list.append(p_smote_nb)

        logreg['iteration'] = i
        logreg1_list.append(logreg)
        unp_smote_logreg['iteration'] = i
        logreg2_list.append(unp_smote_logreg)
        p_smote_logreg['iteration'] = i
        logreg3_list.append(p_smote_logreg)

        print("\nDATA RATIO")
        print('Training data \t{}'.format (Counter(y_train)))
        print('Testing data \t{}'. format(Counter(y_test)))
        print('Resampled training data')
        print('Unproportional \t{}'.format(Counter(y_res_unp)))
        print('Proportional \t{}'.format(Counter(y_res_p)))

        # APPEND ALL y_pred_class LISTS WITH THEIR CORRESPONDING review_id
        nb1_y_list.extend([list(x) for x in zip(test_df['review_id'], nb_y)])
        logreg1_y_list.extend([list(x) for x in zip(test_df['review_id'], logreg_y)])

        nb2_y_list.extend([list(x) for x in zip(test_df['review_id'], unp_nb_y)])
        logreg2_y_list.extend([list(x) for x in zip(test_df['review_id'], unp_logreg_y)])

        nb3_y_list.extend([list(x) for x in zip(test_df['review_id'], p_nb_y)])
        logreg3_y_list.extend([list(x) for x in zip(test_df['review_id'], p_logreg_y)])

        # END OF LOOP

    # SUM ALL THE MATRICES TO GET THE AVERAGE ACCURACY FROM MULTIPLE ITERATIONS
    # MNB RESULTS
    print("MultinomialNB, Unproportional SMOTE + MNB, Proportional SMOTE + MNB")
    nb1_sum = sum_all_matrices(nb1_list)
    nb2_sum = sum_all_matrices(nb2_list)
    nb3_sum = sum_all_matrices(nb3_list)
    # LOGREG RESULTS
    print("Logreg, Unproportional SMOTE + Logreg, Proportional SMOTE + Logreg")
    logreg1_sum = sum_all_matrices(logreg1_list)
    logreg2_sum = sum_all_matrices(logreg2_list)
    logreg3_sum = sum_all_matrices(logreg3_list)

    # CALCULATE REVIEW SCORES AND FILM SCORES
    logreg1_scores = calculate_review_scores(logreg1_y_list)
    logreg1_film_scores = calculate_film_scores(logreg1_scores, df[['review_id', 'asin', 'overall']])

    logreg2_scores = calculate_review_scores(logreg2_y_list)
    logreg2_film_scores = calculate_film_scores(logreg2_scores, df[['review_id', 'asin', 'overall']])

    logreg3_scores = calculate_review_scores(logreg3_y_list)
    logreg3_film_scores = calculate_film_scores(logreg3_scores, df[['review_id', 'asin', 'overall']])

    nb1_scores = calculate_review_scores(nb1_y_list)
    nb1_film_scores = calculate_film_scores(nb1_scores, df[['review_id', 'asin', 'overall']])

    nb2_scores = calculate_review_scores(nb2_y_list)
    nb2_film_scores = calculate_film_scores(nb2_scores, df[['review_id', 'asin', 'overall']])

    nb3_scores = calculate_review_scores(nb3_y_list)
    nb3_film_scores = calculate_film_scores(nb3_scores, df[['review_id', 'asin', 'overall']])

    # nb1_y = pd.DataFrame(list(zip(nb1_y)))
    #
    # nb1_y.groupby('review_id').agg(lambda x: x.tolist())
    # print(nb1_y)
    # # SAVE THE SUM RESULTS
    # sum_df = pd.concat([nb1_sum, nb2_sum, nb3_sum, logreg1_sum, logreg2_sum, logreg3_sum])
    # sum_df.to_csv("/home/lia/Documents/the_project/output/sum.csv")
    #
    # # CONCAT ALL THE DATAFRAMES INSIDE LISTS
    # nb1_df = pd.concat(nb1_list)
    # nb2_df = pd.concat(nb2_list)
    # nb3_df = pd.concat(nb3_list)
    # logreg1_df = pd.concat(logreg1_list)
    # logreg2_df = pd.concat(logreg2_list)
    # logreg3_df = pd.concat(logreg3_list)
    #
    # # SAVE ALL THE CONCATENATED DATAFRAMES TO THEIR OWN CSV FILES
    # nb1_df.to_csv("/home/lia/Documents/the_project/output/nb.csv")
    # nb2_df.to_csv("/home/lia/Documents/the_project/output/unp_smote_nb.csv")
    # nb3_df.to_csv("/home/lia/Documents/the_project/output/p_smote_nb.csv")
    # logreg1_df.to_csv("/home/lia/Documents/the_project/output/logreg.csv")
    # logreg2_df.to_csv("/home/lia/Documents/the_project/output/unp_smote_logreg.csv")
    # logreg3_df.to_csv("/home/lia/Documents/the_project/output/p_smote_logreg.csv")
    # print("file saved")


def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    input_file = '/home/lia/Documents/the_project/dataset/to_use/current/clean_data.csv'

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
