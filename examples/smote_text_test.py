import pandas as pd
import numpy as np
from sklearn import datasets
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

def ratio_dict(old_ratio):
    new_ratio = {}

    for i in range(1,len(old_ratio)):
        new_ratio[i] = old_ratio[i]*2
        if i==2:
            new_ratio[i] = old_ratio[i]*4

    return new_ratio


print("Oversampling")

# input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/6.csv'
# input_file = '/home/lia/Documents/the_project/dataset/output/clean_df.csv'
input_file = '/home/lia/Documents/the_project/dataset/top_10_movies/top_10_clean.csv'

df = pd.read_csv(input_file)

# SPLIT INTO TRAINING AND TESTING

train_df, test_df = train_test_split(df, test_size=0.33)

X_train = train_df['reviewText']
y_train = train_df['overall']
X_test = test_df['reviewText']
y_test = test_df['overall']

print(X_train.head(10))

# VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# OVERSAMPLE WITH SMOTE
# sm = SMOTE(ratio={2:700},random_state=42)
sm = SMOTE(ratio=ratio_dict(Counter(y_train)))
X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

print('Original data {}'.format (Counter(y_train)))
print('Resampled data {}'.format(Counter(y_res)))

X_train_vectorized, y_train = X_res, y_res

# FEATURE SELECTION WITH chi2
chi = SelectKBest(chi2)
X_new = chi.fit_transform(X_train_vectorized, y_train)
print(X_train_vectorized.shape)
print(X_new.shape)

X_train_vectorized = X_new

# VECTORIZE X_test
X_test_vectorized = vectorizer.transform(X_test)

# FIT INTO CLASSIFIER
# clf = MultinomialNB()
clf = linear_model.LogisticRegression()

# TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
clf.fit(X_train_vectorized, y_train)

y_pred_class = clf.predict(X_test_vectorized)

accu = metrics.accuracy_score(y_test, y_pred_class)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

report_matrix = metrics.classification_report(y_test, y_pred_class)
print(report_matrix)


# interesting note to look at
# X_train = vectorizer.fit_transform(data_train.data) #fit_transform on training data
# X_test = vectorizer.transform(data_test.data) #just transform on test data
