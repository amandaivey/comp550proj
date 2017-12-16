'''
@author: Theodore Morley
A set of functions for preprocessing data before moving it into the model
'''

import numpy
from sklearn import utils
from sklearn import model_selection

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

def main():
    X = [[0,0,1],[1,1,0]]
    Y = [0, 1]
    tr, te = shuffle(X, Y, 0.5)
    print(tr)
    print(te)

if __name__=='__main__':
    main()