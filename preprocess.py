'''
@author: Theodore Morley
A set of functions for preprocessing data before moving it into the model
'''

import numpy
from sklearn import utils
from sklearn import model_selection
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

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

def tokenize(data):
    toked = []
    tknzr = TweetTokenizer()
    for d in data:
        toked.append(tknzr[d])
    return toked

def casing(data):
    lower = []
    for d in data:
        lower.append(d.lower())
    return lower

def stops(data):
    unstoppable = []
    for d in data:
        if d not in STOPWORDS:
            unstoppable.append(d)
    return unstoppable

def punctuation(data):
    nopunct = []
    for d in data:
        if d not in string.punctuation:
            nopunct.append(d)
    return nopunct

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

if __name__=='__main__':
    main()
