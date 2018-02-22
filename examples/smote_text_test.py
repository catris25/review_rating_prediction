import pandas as pd
from sklearn import datasets
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from collections import Counter

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

print("Oversampling")

# input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/6.csv'
input_file = '/home/lia/Documents/the_project/dataset/output/clean_df.csv'
df = pd.read_csv(input_file)

X_data = df['reviewText']
y_data = df['overall']

# SPLIT INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# VECTORIZE AND FIT_TRANSFORM THE TRAINING DATA
vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# OVERSAMPLE WITH SMOTE
# sm = SMOTE(ratio={2:700},random_state=42)
sm = SMOTE(ratio='all')
X_res, y_res = sm.fit_sample(X_train_vectorized, y_train)

# OVERSAMPLE WITH NAIVE RANDOM OVERSAMPLE
# ros = RandomOverSampler(random_state=0)
# X_res, y_res = ros.fit_sample(X_train_vectorized, y_train)

print('Original data {}'.format (Counter(y_train)))
print('Resampled data {}'.format(Counter(y_res)))

# VECTORIZE X_test
X_test_vectorized = vectorizer.transform(X_test)

# FIT INTO CLASSIFIER
clf = MultinomialNB()

# TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
clf.fit(X_res, y_res)

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
