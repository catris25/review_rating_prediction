from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np

def vectorize_data(X_train, X_test):
    # VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print(vectorizer.get_feature_names())

    return (X_train_vectorized, X_test_vectorized)

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


input_file = '/home/lia/Documents/the_project/dataset/output/test_data.csv'
df = pd.read_csv(input_file)

train_df, test_df = train_test_split(df, test_size=0.3)

# READ TRAINING DATA AND SEPARATE INTO X AND y
X_train = train_df['reviewText']
y_train = train_df['overall']

# READ TESTING DATA AND SEPARATE INTO X AND y
X_test = test_df['reviewText']
y_test = test_df['overall']

# VECTORIZE THE DATA
X_train_vectorized, X_test_vectorized = vectorize_data(X_train, X_test)
print(X_test)
print(X_test_vectorized[0])

print("Logistic Regression")
logreg, lr_y = classify_logreg_report(X_train_vectorized, y_train, X_test_vectorized, y_test)
print(lr_y)
