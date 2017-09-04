# import random
import pickle

# import nltk
# from nltk.corpus import movie_reviews
# from nltk.classify.scikitlearn import SklearnClassifier
#
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
#
# MNB_classifier = SklearnClassifier(MultinomialNB())
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]


from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Did you ever hear the tragedy of Darth Plagueis The Wise? I thought not. It’s not a story the Jedi would tell you. It’s a Sith legend. Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life… He had such a knowledge of the dark side, he could even keep the ones he cared about from dying."

print(sent_tokenize(example_text))
