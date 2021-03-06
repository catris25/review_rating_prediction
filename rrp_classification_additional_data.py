# LIA RISTIANA 2018
# This file contains the classification step in review-rating-prediction task
# which includes classifications using MNB and Logistic Regression
# and oversampling using SMOTE
# results produced are saved in files

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

import os, time

# SUM ALL MATRICES
def sum_all_matrices(matrix_list, m_name):
    sum_matrix = np.sum([df.iloc[:,0:5].values for df in matrix_list], axis=0)
    print(sum_matrix)
    avg_accu = sum(np.diag(sum_matrix))/sum_matrix.sum()
    print(avg_accu)

    matrix_df = pd.DataFrame(sum_matrix, columns=[1,2,3,4,5])
    matrix_df['name'] = m_name
    matrix_df.index = np.arange(1, len(matrix_df) + 1)
    return matrix_df

# GENERATE A DICTIONARY THAT WILL DECIDE HOW MANY RESAMPLES FOR EACH CLASS SHOULD GET
# USED IN PROPORTIONAL OVERSAMPLING
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

# CLASSIFY THE DATA IN DATAFRAME
def classify_data(train_df, test_df, n_loop):
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
        # train_df, test_df = train_test_split(df, test_size=0.3)

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
        iteration = [i]*len(test_df)
        nb1_y_list.extend([list(x) for x in zip(test_df['review_id'], y_test, nb_y, iteration)])
        logreg1_y_list.extend([list(x) for x in zip(test_df['review_id'], y_test, logreg_y, iteration)])

        nb2_y_list.extend([list(x) for x in zip(test_df['review_id'], y_test, unp_nb_y, iteration)])
        logreg2_y_list.extend([list(x) for x in zip(test_df['review_id'], y_test, unp_logreg_y, iteration)])

        nb3_y_list.extend([list(x) for x in zip(test_df['review_id'], y_test, p_nb_y, iteration)])
        logreg3_y_list.extend([list(x) for x in zip(test_df['review_id'], y_test, p_logreg_y, iteration)])

        # END OF LOOP

    # SUM ALL THE MATRICES TO GET THE AVERAGE ACCURACY FROM MULTIPLE ITERATIONS
    # MNB RESULTS
    print("MultinomialNB, Unproportional SMOTE + MNB, Proportional SMOTE + MNB")
    nb1_sum = sum_all_matrices(nb1_list, "nb")
    nb2_sum = sum_all_matrices(nb2_list, "unp_smote_nb")
    nb3_sum = sum_all_matrices(nb3_list, "p_smote_nb")
    # LOGREG RESULTS
    print("Logreg, Unproportional SMOTE + Logreg, Proportional SMOTE + Logreg")
    logreg2_sum = sum_all_matrices(logreg2_list, "unp_smote_logreg")
    logreg1_sum = sum_all_matrices(logreg1_list, "logreg")
    logreg3_sum = sum_all_matrices(logreg3_list, "p_smote_logreg")

    # SAVE ALL RESULTS TO FILES
    output_file = "/home/lia/Dropbox/output/additional_dataset/"+time.strftime("%Y-%m-%d")+"/"
    os.makedirs("%s"%output_file, exist_ok=True)
    os.makedirs("%smatrices"%output_file, exist_ok=True)
    os.makedirs("%sdata"%output_file, exist_ok=True)

    # SAVE THE SUM RESULTS
    sum_df = pd.concat([nb1_sum, nb2_sum, nb3_sum, logreg1_sum, logreg2_sum, logreg3_sum])
    sum_df.to_csv(output_file+"sum.csv")

    # SAVE CONFUSION MATRICES OF ALL CLASSIFICATIONS
    # CONCAT ALL THE DATAFRAMES INSIDE LISTS
    nb1_df = pd.concat(nb1_list)
    nb2_df = pd.concat(nb2_list)
    nb3_df = pd.concat(nb3_list)
    logreg1_df = pd.concat(logreg1_list)
    logreg2_df = pd.concat(logreg2_list)
    logreg3_df = pd.concat(logreg3_list)

    # SAVE ALL THE CONCATENATED DATAFRAMES TO THEIR OWN CSV FILES
    folder_name = output_file+"matrices/"
    nb1_df.to_csv(folder_name+"nb.csv")
    nb2_df.to_csv(folder_name+"unp_smote_nb.csv")
    nb3_df.to_csv(folder_name+"p_smote_nb.csv")
    logreg1_df.to_csv(folder_name+"logreg.csv")
    logreg2_df.to_csv(folder_name+"unp_smote_logreg.csv")
    logreg3_df.to_csv(folder_name+"p_smote_logreg.csv")
    print("confusion matrices all saved")

    # SAVE REVIEW DATA WITH THEIR PREDICTED CLASSES
    folder_name = output_file +"data/"
    col_list = ['review_id', 'actual', 'prediction', 'iteration']
    pd.DataFrame(nb1_y_list, columns=col_list).to_csv(folder_name+"nb.csv", index=False)
    pd.DataFrame(nb2_y_list, columns=col_list).to_csv(folder_name+"unp_smote_nb.csv", index=False)
    pd.DataFrame(nb3_y_list, columns=col_list).to_csv(folder_name+"p_smote_nb.csv", index=False)
    pd.DataFrame(logreg1_y_list, columns=col_list).to_csv(folder_name+"logreg.csv", index=False)
    pd.DataFrame(logreg2_y_list, columns=col_list).to_csv(folder_name+"unp_smote_logreg.csv", index=False)
    pd.DataFrame(logreg3_y_list, columns=col_list).to_csv(folder_name+"p_smote_logreg.csv", index=False)
    print("result dataframes all saved")

    # SAVE ORIGINAL DATAFRAME FOR LATER REFERENCE
    test_df.to_csv(output_file+"test_df.csv", index=False)

def main():
    # input_file = '/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv'
    # READ THE TRAINING DATA (CLEANED DATA)
    input_file = "/home/lia/Dropbox/output/preprocessed.csv"
    prep_df = pd.read_csv(input_file)

    # READ THE TESTING DATA (RAW/UNCLEANED DATA)
    input_file = "/home/lia/Dropbox/output/additional_dataset/preprocessed.csv"
    raw_df = pd.read_csv(input_file)

    # PRINT STATS OF DATA
    n_reviews = len(prep_df)
    n_movies = len(prep_df['asin'].value_counts())
    print(" %d reviews of %d movies"%(n_reviews, n_movies))
    print(prep_df['overall'].value_counts().sort_index())

    # PRINT STATS OF DATA
    n_reviews = len(raw_df)
    n_movies = len(raw_df['asin'].value_counts())
    print(" %d reviews of %d movies"%(n_reviews, n_movies))
    print(raw_df['overall'].value_counts().sort_index())

    n_loop = 1
    classify_data(prep_df, raw_df, n_loop)


if __name__ == "__main__":
    main()
