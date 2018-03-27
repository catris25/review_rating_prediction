import pandas as pd
import numpy as np
from sklearn import datasets
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from collections import Counter

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

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


print("Oversampling")

input_file = "/home/lia/Documents/the_project/dataset/to_use/current/top_30_clean.csv"
# input_file = '/home/lia/Documents/the_project/dataset/output/temp.csv'
# input_file = '/home/lia/Documents/the_project/dataset/top_10_movies/top_10_clean.csv'

df = pd.read_csv(input_file)

n_reviews = len(df)
n_movies = len(df['asin'].value_counts())
print(" %d reviews of %d movies"%(n_reviews, n_movies))
print(df['overall'].value_counts().sort_index())

# SPLIT INTO TRAINING AND TESTING

train_df, test_df = train_test_split(df, test_size=0.25)

X_train = train_df['reviewText']
y_train = train_df['overall']
X_test = test_df['reviewText']
y_test = test_df['overall']

print(X_train.head(10))

# VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(min_df=5)
X_train_vectorized = vectorizer.fit_transform(X_train)

# VECTORIZE X_test
X_test_vectorized = vectorizer.transform(X_test)

# OVERSAMPLE WITH SMOTE
# sm = SMOTE(ratio={2:700},random_state=42)
sm = SMOTE(ratio=ratio_dict(Counter(y_train)))
X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

print('Original data {}'.format (Counter(y_train)))
print('Resampled data {}'.format(Counter(y_res)))

X_train_vectorized, y_train = X_res, y_res

# FEATURE SELECTION
# selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X_new = selector.fit_transform(X_train_vectorized)
# selector = SelectKBest(chi2, 6500)
# X_new = selector.fit_transform(X_train_vectorized, y_train)
#
# print(X_train_vectorized.shape)
# print(X_new.shape)
#
# X_train_vectorized = X_new
# X_test_vectorized = selector.transform(X_test_vectorized)


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

print("\nMISCLASSIFIED")
incorrect = np.where(y_test != y_pred_class)
test_df = test_df.assign(predicted=y_pred_class)

row_ids = test_df[test_df["overall"] != test_df['predicted']].index
new_df = test_df.loc[row_ids][['reviewText', 'overall', 'predicted']]
print(new_df)
# new_df.to_csv('/home/lia/Documents/the_project/dataset/to_use/current/top_30_incorrect.csv')


# incorrect_asin = df.ix[incorrect][['reviewText', 'overall']]
# print(incorrect_asin)
# interesting note to look at
# X_train = vectorizer.fit_transform(data_train.data) #fit_transform on training data
# X_test = vectorizer.transform(data_test.data) #just transform on test data
