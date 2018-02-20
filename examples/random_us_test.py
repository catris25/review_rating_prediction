from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

import pandas as pd

# import warnings
# import sklearn.exceptions
# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/30percent/3.csv'
df = pd.read_csv(input_file)

# categories = list(set(df['overall']))
X_data = df['reviewText']
y_data = df['overall']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

print('Training class distributions summary: {}'.format(Counter(y_train)))
print('Test class distributions summary: {}'.format(Counter(y_test)))

pipe = make_pipeline_imb(TfidfVectorizer(), RandomUnderSampler(), MultinomialNB())
# pipe = make_pipeline(CountVectorizer(), MultinomialNB())
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(classification_report_imbalanced(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
