import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import word_tokenize
from nltk import TweetTokenizer
from nltk import NaiveBayesClassifier
from nltk.util import bigrams
from nltk.corpus import stopwords
import numpy as np
from nltk.corpus import PlaintextCorpusReader
import string
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import PorterStemmer
import codecs
from sklearn import datasets
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn import cross_validation

csv_file = "/Users/amandaivey/PycharmProjects/comp550proj/apple_data_2.csv"

NUMBER_OF_FEATS = 500
STOPWORDS = set(stopwords.words('english'))

#Over whole dataset
def tokenize(data):
    toked = []
    tknzr = TweetTokenizer()
    for d in data:
        toked.append(tknzr.tokenize(d))
    return toked

def stops(data):
    unstoppable = []
    for tweet in data:
        t=[]
        for word in tweet:
            if word not in STOPWORDS:
                t.append(word)
        unstoppable.append(t)
    return unstoppable

#Takes tokenized dataset
def casing(data):
    lower = []
    for tweet in data:
        t = []
        for word in tweet:
            t.append(word.lower())
        lower.append(t)
    return lower

#Takes tokenized
def punctuation(data):
    nopunct = []
    for tweet in data:
        t=[]
        for word in tweet:
            if word not in string.punctuation:
                t.append(word)
        nopunct.append(t)
    return nopunct

def stem_all(data):
    stemmed = []
    stemmer = PorterStemmer()
    for tweet in data:
        t=[]
        for word in tweet:
            t.append(stemmer.stem(word))
        stemmed.append(t)
    return stemmed

FUNCTIONS = [tokenize, casing, punctuation, stem_all, stops]

def setup_corpus(data):
    for f in FUNCTIONS:
        data = f(data)
    representation_dict = {}
    filtered_list = []
    for tweet in data:
        for item in tweet:
            if item not in representation_dict:
                representation_dict[item] = 1
            else:
                representation_dict[item] += 1

    for key, value in sorted(representation_dict.iteritems(), key=lambda (k, v): (v, k)):
        filtered_list.append((key, value))

    filtered_list.reverse()

    return {i[0]: i[1] for i in filtered_list[:NUMBER_OF_FEATS]}

def preprocess_features(data, labels, corpus_dict):
    data_vector = []
    for f in FUNCTIONS:
        data = f(data)
    for review in data:
        if len(review) > 0:
            data_vector.append(review)
    return (zip(data_vector, labels))

def generate_data_vectors(pre_processed_data, corpus_dict):
    vector_list = []
    for review in pre_processed_data:
        temp_vector = np.zeros(NUMBER_OF_FEATS)
        for word in review[0]:
            if word in corpus_dict:
                temp_vector[corpus_dict.keys().index(word)] = 1
        try:
            vector_list.append((temp_vector, (review[1])))
        except ValueError:
            continue

    return vector_list

def get_accuracy(correct, predictions):
    total = 0
    total_correct = 0
    i = 0
    while i < len(correct):
        total += 1
        if correct[i] == predictions[i]:
            total_correct += 1
        i += 1
    return float(total_correct)/total

def test_model(classifier):
    df = pd.read_csv(csv_file, sep=',', header=None)
    x = np.array(df.ix[:, 0])
    y = np.array(df.ix[:, 1])
    corpus_dict = setup_corpus(x)
    preproc_data = preprocess_features(x, y, corpus_dict)
    vector = generate_data_vectors(preproc_data, corpus_dict)
    x_data = []
    y_data = []
    for data in vector:
        x_data.append(data[0])
        y_data.append(data[1])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    if classifier == "svm":
        clf = svm.SVC(kernel='linear')
    else:
        clf = BernoulliNB()

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = (get_accuracy(y_test, predictions))
    print "Average " + classifier + " accuracy: {}".format(accuracy)
    return accuracy

def main():
    test_model("svm")
    test_model("bayes")

if __name__=='__main__':
    main()
