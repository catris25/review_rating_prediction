import pandas as pd
from sklearn import datasets
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from collections import Counter

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

# input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/1.csv'
input_file = '/home/lia/Documents/the_project/dataset/output/clean_df.csv'
df = pd.read_csv(input_file)

X_data = df['reviewText']
y_data = df['overall']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

# count_vector= CountVectorizer(ngram_range=(1,1))
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(X_train)
x_train_counts.toarray()
# tf_transformer = TfidfTransformer().fit(x_train_counts)
# X_train_tf = tf_transformer.transform(x_train_counts)
X_train_tf = TfidfTransformer().fit_transform(x_train_counts)
# X_train_tf.toarray()
# tfidf_transformer = TfidfTransformer()

# sure this can be made more efficient? wth is this
# tf_transformer = TfidfTransformer().fit(x_train_counts)
# X_train_tf = tf_transformer.transform(x_train_counts)
# X_train_tf = TfidfTransformer().fit_transform(x_train_counts)
# X_train_tf.toarray()

sm = SMOTE()
X_res, y_res = sm.fit_sample(x_train_counts, y_train)
X_res, y_res=sm.fit_sample(X_res,y_res)
print('Original data {}'.format (Counter(y_train)))
print('Resampled data {}'.format(Counter(y_res)))

# VECTORIZE X_test
X_test = vectorizer.transform(X_test)

# FIT INTO CLASSIFIER
clf = MultinomialNB()
# clf = linear_model.LogisticRegression()
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,7))

# TRAIN THE CLASSIFIER WITH AVAILABLE TRAINING DATA
clf.fit(X_res, y_res)

y_pred_class = clf.predict(X_test)

accu = metrics.accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

report_matrix = metrics.classification_report(y_test, y_pred_class)
print(report_matrix)


# interesting note to look at
# X_train = vectorizer.fit_transform(data_train.data) #fit_transform on training data
# X_test = vectorizer.transform(data_test.data) #just transform on test data
