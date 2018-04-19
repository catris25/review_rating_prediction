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

    # report_matrix = metrics.classification_report(y_test, y_pred_class)
    # print(report_matrix)

    return(df_conf_matrix)

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

    # report_matrix = metrics.classification_report(y_test, y_pred_class)
    # print(report_matrix)

    return(df_conf_matrix)

# VECTORIZE DATA INTO A SPARSE MATRIX
def vectorize_data(X_train, X_test):
    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return (X_train_vectorized, X_test_vectorized)

def classify_data(df,i):
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
    nb = classify_nb_report(X_train_vectorized, y_train, X_test_vectorized, y_test)
    print("Logistic Regression")
    logreg = classify_logreg_report(X_train_vectorized, y_train, X_test_vectorized, y_test)

    X_res_unp, y_res_unp = oversample_unproportional(X_train_vectorized, y_train)
    print("Unproportional SMOTE + MNB")
    unp_smote_nb = classify_nb_report(X_res_unp, y_res_unp, X_test_vectorized, y_test)
    print("Unproportional SMOTE + LogReg")
    unp_smote_logreg = classify_logreg_report(X_res_unp, y_res_unp, X_test_vectorized, y_test)

    X_res_p, y_res_p = oversample_proportional(X_train_vectorized, y_train)
    print("Proportional SMOTE + MNB")
    p_smote_nb = classify_nb_report(X_res_p, y_res_p, X_test_vectorized, y_test)
    print("Proportional SMOTE + LogReg")
    p_smote_logreg = classify_logreg_report(X_res_p, y_res_p, X_test_vectorized, y_test)

    # nb_df.append(pd.Series([np.nan]),ignore_index=True)[nb_df.columns]
    # nb_df = pd.concat(nb_list)
    # logreg_df = pd.concat(logreg_list)

    print("\nDATA RATIO")
    print('Training data \t{}'.format (Counter(y_train)))
    print('Testing data \t{}'. format(Counter(y_test)))
    print('Resampled training data')
    print('Unproportional \t{}'.format(Counter(y_res_unp)))
    print('Proportional \t{}'.format(Counter(y_res_p)))

    return (nb, logreg, unp_smote_nb, unp_smote_logreg, p_smote_nb, p_smote_logreg)

def main():
    input_file = '/home/lia/Documents/the_project/dataset/to_use/current/clean_data.csv'
    # input_file = '/home/lia/Documents/the_project/dataset/output/without_five.csv'

    prep_df = pd.read_csv(input_file)

    # PRINT STATS OF DATA
    n_reviews = len(prep_df)
    n_movies = len(prep_df['asin'].value_counts())
    print(" %d reviews of %d movies"%(n_reviews, n_movies))
    print(prep_df['overall'].value_counts().sort_index())

    nb1_list = []
    logreg1_list = []

    nb2_list = []
    logreg2_list = []

    nb3_list = []
    logreg3_list = []

    # CLASSIFY THE DATA AND REPEAT IT 30 TIMES
    for i in range(0,10):
        print("**ITERATION-%d**"%i)
        nb1, logreg1, nb2, logreg2, nb3, logreg3 = classify_data(prep_df,i)

        nb1.insert(loc=0, column='iteration', value=i)
        logreg1.insert(loc=0, column='iteration', value=i)
        nb1_list.append(nb1)
        logreg1_list.append(logreg1)

        nb2.insert(loc=0, column='iteration', value=i)
        logreg2.insert(loc=0, column='iteration', value=i)
        nb2_list.append(nb2)
        logreg2_list.append(logreg2)

        nb3.insert(loc=0, column='iteration', value=i)
        logreg3.insert(loc=0, column='iteration', value=i)
        nb3_list.append(nb3)
        logreg3_list.append(logreg3)

    nb1_df = pd.concat(nb1_list)
    logreg1_df = pd.concat(logreg1_list)

    nb2_df = pd.concat(nb2_list)
    logreg2_df = pd.concat(logreg2_list)

    nb3_df = pd.concat(nb3_list)
    logreg3_df = pd.concat(logreg3_list)

    nb1_df.to_csv("/home/lia/Documents/the_project/output/nb.csv",index=False)
    logreg1_df.to_csv("/home/lia/Documents/the_project/output/logreg.csv", index=False)

    nb2_df.to_csv("/home/lia/Documents/the_project/output/unp_smote_nb.csv",index=False)
    logreg2_df.to_csv("/home/lia/Documents/the_project/output/unp_smote_logreg.csv", index=False)

    nb3_df.to_csv("/home/lia/Documents/the_project/output/p_smote_nb.csv",index=False)
    logreg3_df.to_csv("/home/lia/Documents/the_project/output/p_smote_logreg.csv", index=False)
    print("file saved")

if __name__ == "__main__":
    main()
