import pandas as pd
import numpy as np
import re, string
import sys

import nltk
import scipy as sc

from nltk import bigrams, trigrams, FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.classify import NaiveBayesClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm, grid_search
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.cross_validation import train_test_split

# ---- PREPROCESSING STEP ----
def preprocess(df):
    # STEP 1. punctuation removal, lowering capital letters, and tokenization
    df_removal = []
    ps_token = PunktSentenceTokenizer()
    for review in df['reviewText']:
        temp = re.sub("[^a-zA-Z']", " ", str(review))
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
    sw = set(stopwords.words("english"))
    stop_words = sw.union(["film", "movie", "movies", "films"])

    # print(stop_words)
    # sys.exit("ok")
    for review in df['reviewText']:
        filtered_text = [word for word in review if not word in stop_words]
        df_filtering.append(filtered_text)

    df['reviewText'] = df_filtering
    print(df['reviewText'].head(5))
    print(sum([len(r) for r in df['reviewText']]))

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

    # STEP 3. Stemming
    df_stemming = []
    ss = SnowballStemmer('english')
    ps = PorterStemmer()
    for review in df['reviewText']:
        stemmed_text = [ss.stem(word) for word in review]
        df_stemming.append(stemmed_text)

    df['reviewText'] = df_stemming
    print(df['reviewText'].head(5))

    return df

def sep_to_x_y(df):
    reviews = df['reviewText']
    X_data = [(',').join(review) for review in reviews ]
    y_data = df['overall']
    return X_data, y_data

def find_common_words(df):
    all_words =[]
    for review in df['reviewText']:
        for word in review:
            all_words.append(word)

    fdist = FreqDist(all_words)
    most_common = fdist.most_common(10)

    return most_common

def find_bigrams(df):
    list_bigrams = list(bigrams(all_words))
    fdist = FreqDist(list_bigrams)
    most_common = fdist.most_common(10)

def ent(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy

# input_file = '/home/lia/Documents/the_project/dataset/top_30_movies/helpful/30percent/3.csv'
input_file = '/home/lia/Documents/the_project/dataset/to_use/helpfulness/samples/10percent/27.csv'
# input_file = "/home/lia/Documents/the_project/dataset/clean_airline_sentiments.csv"
# input_file = '/home/lia/Documents/the_project/dataset/top_30_movies/top_30.csv'
orig_df = pd.read_csv(input_file)
print(orig_df['overall'].value_counts())
df = preprocess(orig_df)

# df.to_csv("/home/lia/Documents/the_project/dataset/clean_df.csv", index=False, sep=',', encoding='utf-8')
#
ratings = [1,2,3,4,5]

for rtg in ratings:
    temp = df.loc[df['overall']==rtg]
    # print(temp.head(5))
    common_words = find_common_words(temp)

    print(rtg)
    print("---------")
    print(common_words)
#
# sys.exit("ok")

# CALCULATING ACCURACY
X_data, y_data = sep_to_x_y(df)

# vect = TfidfVectorizer(binary=True, min_df=5, ngram_range=(1,3))
vect = CountVectorizer(binary=False, min_df=5, ngram_range=(1,2))
X_dtm = vect.fit_transform(X_data)

print(vect.get_feature_names())

print(X_dtm.toarray().shape)

mnb = MultinomialNB()
knn = KNeighborsClassifier(n_neighbors=9)
svr = svm.SVC(kernel='rbf',C=15.0,gamma=0.01)
logreg = linear_model.LogisticRegression()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,4,6), random_state=12)
# best mlp [2,4] [3,4,6]

clf = mnb
X_train, X_test, y_train, y_test = train_test_split(X_dtm, y_data, test_size=0.3, random_state=123)

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

accu = accuracy_score(y_pred_class, y_test)
print(accu)


# reviews = []
# for data in df:
#     rev = df[['reviewText', 'overall']]
#     reviews.append(rev)
# reviews = list(df[['reviewText', 'overall']].values.flatten())
# reviews = df[['reviewText', 'overall']].to_records(index=False).tolist()
# reviews = df[['reviewText', 'overall']].to_dict()
# print(reviews)

# classifier = NaiveBayesClassifier.train(reviews)
# classifier.show_most_informative_features()

conf_matrix = metrics.confusion_matrix(y_test,y_pred_class)
print(conf_matrix)

class_labels = [1,2,3,4,5]
feature_names = vect.get_feature_names()
for i, class_label in enumerate(class_labels):
    top10 = np.argsort(clf.coef_[i])[-20:]
    print("%s: %s" % (class_label,
          ", ".join(feature_names[j] for j in top10)))
