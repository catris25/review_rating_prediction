import pandas as pd
import numpy as np
import re, string
import sys

import nltk
import nltk

from nltk import bigrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, grid_search
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/20percent/1.csv'
df = pd.read_csv(input_file)
# df = df.tail(200)

# new_df = pd.DataFrame()
# levels = [1,2,3,4,5]
# for lvl in levels:
#     temp = df.loc[df['overall']==lvl].sample(n=100)
#     new_df = pd.concat([new_df,temp])

# print(new_df['overall'].value_counts())
print(df['overall'].value_counts())
# df = new_df

# ---- PREPROCESSING STEP ----
def preprocess(df):
    # STEP 1. punctuation removal, lowering capital letters, and tokenization
    df_removal = []
    ps_token = PunktSentenceTokenizer()
    for review in df['reviewText']:
        temp = re.sub("[^a-zA-Z'-]", " ", str(review))
        temp = temp.replace("'", "")
        temp = temp.lower()
        temp = word_tokenize(temp)
        # print(temp)
        # sys.exit("ok")
        df_removal.append(temp)

    df['reviewText'] = df_removal
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    # STEP 2. Stop words removal/filtering
    df_filtering = []
    stop_words = set(stopwords.words("english"))
    for review in df['reviewText']:
        filtered_text = [word for word in review if not word in stop_words]
        df_filtering.append(filtered_text)

    df['reviewText'] = df_filtering
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    # STEP 3. Stemming
    # df_stemming = []
    # ss = SnowballStemmer('english')
    # ps = PorterStemmer()
    # for review in df['reviewText']:
    #     stemmed_text = [ps.stem(word) for word in review]
    #     df_stemming.append(stemmed_text)
    #
    # df['reviewText'] = df_stemming
    # print(df['reviewText'].head(5))

    #STEP 4. Lemmatization
    df_lemma = []
    wnl = WordNetLemmatizer()

    for review in df['reviewText']:
        lemmatized_text = [wnl.lemmatize(word, 'v') for word in review]
        lemmatized_text = [wnl.lemmatize(word, 'n') for word in lemmatized_text]
        lemmatized_text = [wnl.lemmatize(word, 'a') for word in lemmatized_text]
        df_lemma.append(lemmatized_text)

    df['reviewText'] = df_lemma
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

    return df

def sep_to_x_y(df):
    reviews = df['reviewText']
    X_data = [(',').join(review) for review in reviews ]
    y_data = df['overall']
    return X_data, y_data

# review_set = df['reviewText'].iloc[3]
# print(len(review_set))
# print(len(sorted(set(review_set))))
# print(list(bigrams(review_set)))
#
# sys.exit("ok")

df = preprocess(df)

# CALCULATING ACCURACY
X_train, y_train = sep_to_x_y(df)

# print(X_train[0])

vect = TfidfVectorizer(binary=True, min_df=0.035, max_df=0.25)
X_train_dtm = vect.fit_transform(X_train)

print(X_train_dtm.toarray().shape)

mnb = MultinomialNB()
svr = svm.SVC(kernel='rbf',C=10.0,gamma=0.1)
logreg = linear_model.LogisticRegression()

y_pred_class = cross_val_predict(mnb, X_train_dtm, y_train, cv=10)
y_test = y_train

accu = accuracy_score(y_pred_class, y_test)
print(accu)

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

# train_set = X_train
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# most_informative = classifier.show_most_informative_features(5)
#
# print(most_informative)
