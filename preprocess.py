'''
@author: Theodore Morley
A set of functions for preprocessing data before moving it into the model
TODO IDEAS:
    Remove URLS
'''

import string
import numpy
from sklearn import utils
from sklearn import model_selection
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
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

#def stem(data):
#    stemmed = []
#    for d in data:
#        stemmed.append()

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
    d = ['Hi bob builder', 'THANKS OBUMMER !!!!!']
    t = [0, -1]
    a, b = full_preprocess(d, t, [tokenize, casing, stops, punctuation], .5)
    print(a)
    print(b)

if __name__=='__main__':
    main()
