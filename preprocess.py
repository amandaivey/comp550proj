'''
@author: Theodore Morley
A set of functions for preprocessing data before moving it into the model
TODO IDEAS:
    Remove URLS
'''

import string
import numpy
import gensim
import torch
from sklearn import utils
from sklearn import model_selection
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import stem
STOPWORDS = set(stopwords.words('english'))



def shuffle(data, targets, size_test):
    shuffler = model_selection.ShuffleSplit(n_splits = 1, test_size=size_test)
    out = shuffler.split(data, y=targets)
    test = []
    test_targ = []
    train = []
    train_targ = []
    for train_indices, test_indices in out:
        for train_index in train_indices:
            train.append(data[train_index])
            train_targ.append(targets[train_index])
        for test_index in test_indices:
            test.append(data[test_index])
            test_targ.append(targets[test_index])
    return ((train, train_targ), (test, test_targ))

#Over whole dataset
def tokenize(data):
    toked = []
    tknzr = TweetTokenizer()
    for d in data:
        toked.append(tknzr.tokenize(d))
    return toked

#Takes tokenized dataset
def casing(data):
    lower = []
    for tweet in data:
        t = []
        for word in tweet:
            t.append(word.lower())
        lower.append(t)
    return lower

#Takes tokenized dataset
def stops(data):
    unstoppable = []
    for tweet in data:
        t=[]
        for word in tweet:
            if word not in STOPWORDS:
                t.append(word)
        unstoppable.append(t)
    return unstoppable

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
    stemmer = stem.PorterStemmer()
    for tweet in data:
        t=[]
        for word in tweet:
            t.append(stemmer.stem(word))
        stemmed.append(t)
    return stemmed

#takes tokenized
def get_max_len(data):
    max_len = 0
    for tweet in data():
        len = 0
        for word in tweet:
            len += 1
        if len > max_len:
            max_len = len
    return max_len

#takes tokenized
def pad(data):
    start = "<start>"
    end = "<end>"
    pad = "<pad>"
    max_len = get_max_len(data)
    padded_tweets = []
    for tweet in data:
        length = len(tweet)
        add_start = tweet.insert(0, start)
        padded_tweet = add_start
        if length < max_len + 2:
            while length < max_len + 2:
                padded_tweet = padded_tweet + " " + pad
                length += 1
            padded_tweet = padded_tweet + " " + end
        else:
            padded_tweet = padded_tweet + " " + end
        padded_tweets.append(padded_tweet)
    return padded_tweets

#takes tokenized data
def setup_model(data):
    model = gensim.models.Word2Vec(data, min_count=1, size=100, workers=4)
    return model

def tensorize(model, data):
    tensors = []
    max_length = 0
    for tweet in data:
        V_data = []
        for word in tweet:
            V_data.append(torch.Tensor(model.wv[word]))
        tensors.append(V_data)
    return tensors


'''
Input:
    dataset: Iterable containing all the data to be trained on
    targets: Indexed iterable matching ratings/targets to the datapoints in dataset
output: The output of shuffling the dataset and targets into a train and test set determined by testsize
'''
def full_preprocess(dataset, targets, functions, testsize):
    for function in functions:
        dataset = function(dataset)
    return shuffle(dataset, targets, testsize)

def main():
    X = [[0,0,1],[1,1,0]]
    Y = [0, 1]
    tr, te = shuffle(X, Y, 0.5)
    print(tr)
    print(te)
    d = ['Hi bob builder I am thinking', 'THANKS OBUMMER!!!!! <pad>']
    t = [0, -1]
    a, b = full_preprocess(d, t, [tokenize, casing, stops, punctuation, stem_all], .5)
    print(a)
    print(b)
    stemmer = stem.PorterStemmer()
    print(stemmer.stem('Thinking'))

if __name__=='__main__':
    main()
