'''
@author: Theodore Morley
A set of functions for preprocessing data before moving it into the model
TODO IDEAS:
    Remove URLS
'''

import string
import numpy as np
import gensim
import torch
from sklearn import utils
from sklearn import model_selection
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import stem
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
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
    for tweet in data:
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
        tweet.insert(0, start)
        padded_tweet = tweet
        if length < max_len:
            while length < max_len:
                padded_tweet.append(pad)
                length += 1
            padded_tweet.append(end)
        else:
            padded_tweet.append(end)
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
    #print(dataset)
    for function in functions:
        dataset = function(dataset)
    #print(dataset)
    #m = setup_model(dataset)
    #tensorized_data = tensorize(m, dataset)
    return shuffle(dataset, targets, testsize)

def word2ix(all_samples):
    translator = dict()
    for sentence in all_samples:
        for word in sentence:
            if word not in translator:
                translator[word] = len(translator)
    return translator

def encodeSample(sample, embed_dict):
    out = []
    for word in sample:
        out.append(embed_dict[word])
    return out

# Testing NN
def basic_nn(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(get_max_len(trainSamples), get_max_len(testSamples))
    padded = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    # define the model
    model = Sequential()
    model.add(Embedding(len(word_to_index), embed_size, input_length=maximum_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # fit model
    trainTags=np.asarray(trainTags)
    model.fit(padded, trainTags, epochs=epoc, verbose=0)
    # prepare test data
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_test = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # test
    score = model.evaluate(padded, trainTags, verbose=0)#padded_test, testTags, verbose=0)
    return score

def main():
    X = [[0,0,1],[1,1,0]]
    Y = [0, 1]
    tr, te = shuffle(X, Y, 0.5)
    #print(tr)
    #print(te)
    d = ['Hi bob builder I am thinking', 'THANKS OBUMMER!!!!!']
    t = [0, -1]
    a, b = full_preprocess(d, t, [tokenize, casing, stops, punctuation, stem_all], .5)
    print(len(a[0][0]))
    print(len(b[0][0]))
    print(basic_nn(a[0], a[1], b[0], b[1], 12, 300))


if __name__=='__main__':
    main()
