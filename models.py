'''
@author: Theodore Morley
Model setups
'''
import retrieve_data
from sklearn import svm
from sklearn import metrics
import preprocess
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras import callbacks

index2rating = {0:-1, 1:0, 2:1}

def init_baseline(data, targets):
    baseline = svm.SVC()
    baseline.fit(data, targets)
    return baseline


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


def basic_nn(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(preprocess.get_max_len(trainSamples), preprocess.get_max_len(testSamples))
    padded = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    # define the model
    model = Sequential()
    model.add(Embedding(len(word_to_index), embed_size, input_length=maximum_length))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])
    # fit model
    trainTags=np.asarray(trainTags)
    earlystop = callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=10)
    model.fit(padded, trainTags, epochs=epoc, verbose=0, callbacks=[earlystop])
    # prepare test data
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_test = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # test
    predictions = model.predict(padded_test)
    predictions[predictions>=0.5] = 1
    predictions[predictions<0.5] = -1
    confusion = metrics.confusion_matrix(testTags, predictions)
    #score = model.evaluate(padded_test, testTags, verbose=1)
    return confusion

def tern_basic(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(preprocess.get_max_len(trainSamples), preprocess.get_max_len(testSamples))
    padded = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    # define the model
    model = Sequential()
    model.add(Embedding(len(word_to_index), embed_size, input_length=maximum_length))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    # fit model
    trainTags=np.asarray(trainTags)
    earlystop = callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=10)
    model.fit(padded, trainTags, epochs=epoc, verbose=0, callbacks=[earlystop])
    # prepare test data
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_test = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # test
    predictionProbs = model.predict(padded_test)
    predictions = []
    for p in predictionProbs:
        p = p.tolist()
        maxval = max(p)
        predictions.append(index2rating[p.index(maxval)])
    trueTags = []
    for t in testTags:
        maxval = max(t)
        trueTags.append(index2rating[t.index(maxval)])
    confusion = metrics.confusion_matrix(trueTags, predictions)
    #score = model.evaluate(padded_test, testTags, verbose=1)
    return confusion

def convNN(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # prep data
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(preprocess.get_max_len(trainSamples), preprocess.get_max_len(testSamples))
    padded_s = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_t = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # build model
    model = Sequential()
    model.add(Embedding(len(word_to_index), embed_size, input_length=maximum_length))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # test model
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    earlystop = callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=3)
    model.fit(padded_s, trainTags, batch_size=16, epochs=epoc, 
            callbacks=[earlystop])
    #score = model.evaluate(padded_t, testTags, batch_size=16)
    predictions = model.predict(padded_t)
    predictions[predictions>=0.5] = 1
    predictions[predictions<0.5] = -1
    #print(testTags)
    #print(predictions)
    confusion = metrics.confusion_matrix(testTags, predictions)
    return confusion

def tern_convNN(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # prep data
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(preprocess.get_max_len(trainSamples), preprocess.get_max_len(testSamples))
    padded_s = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_t = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # build model
    model = Sequential()
    model.add(Embedding(len(word_to_index), embed_size, input_length=maximum_length))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # test model
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    earlystop = callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=3)
    model.fit(padded_s, trainTags, batch_size=16, epochs=epoc,
            callbacks=[earlystop])
    #score = model.evaluate(padded_t, testTags, batch_size=16)
    predictionProbs = model.predict(padded_t)
    predictions = []
    for p in predictionProbs:
        p = p.tolist()
        maxval = max(p)
        predictions.append(index2rating[p.index(maxval)])
    trueTags = []
    for t in testTags:
        maxval = max(t)
        trueTags.append(index2rating[t.index(maxval)])
    confusion = metrics.confusion_matrix(trueTags, predictions)
    return confusion

def lstm(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # prep data
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(preprocess.get_max_len(trainSamples), preprocess.get_max_len(testSamples))
    padded_s = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_t = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post') 
    # Make model
    model = Sequential()
    model.add(Embedding(len(word_to_index), output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    earlystop = callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=3)
    model.fit(padded_s, trainTags, batch_size=16, epochs=epoc, callbacks=[earlystop])
    #score = model.evaluate(padded_t, testTags, batch_size=16)
    predictions = model.predict(padded_t)
    predictions[predictions>=0.5] = 1
    predictions[predictions<0.5] = -1
    confusion = metrics.confusion_matrix(testTags, predictions)
    return confusion

def tern_lstm(trainSamples, trainTags, testSamples, testTags, embed_size, epoc):
    # prep data
    # First set up word to index encoding
    word_to_index = word2ix(trainSamples)
    # Create encoded list of samples
    encoded_samples = [encodeSample(s, word_to_index) for s in trainSamples]
    # pad all samples
    maximum_length = max(preprocess.get_max_len(trainSamples), preprocess.get_max_len(testSamples))
    padded_s = pad_sequences(encoded_samples, maxlen = maximum_length, padding = 'post')
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_t = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # Make model
    model = Sequential()
    model.add(Embedding(len(word_to_index), output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    earlystop = callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=3)
    model.fit(padded_s, trainTags, batch_size=16, epochs=epoc, callbacks=[earlystop])
    #score = model.evaluate(padded_t, testTags, batch_size=16)
    predictionProbs = model.predict(padded_t)
    predictions = []
    for p in predictionProbs:
        p = p.tolist()
        maxval = max(p)
        predictions.append(index2rating[p.index(maxval)])
    trueTags = []
    for t in testTags:
        maxval = max(t)
        trueTags.append(index2rating[t.index(maxval)])
    confusion = metrics.confusion_matrix(trueTags, predictions)
    return confusion


def main():
    #tweet_dict = retrieve_data.open_csv()
    #rating_dict = retrieve_data.get_rating('2download/gold/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
    #for id in tweet_dict.keys():
    #    if id in rating_dict:
    #        print("in it fam")
    alldata = retrieve_data.genLists('diffsep-mydata.csv')
    tweets = alldata[0]
    ratingsOne = alldata[1]
    # Set ternary ratings
    ratings=[]
    for rating in ratingsOne:
        if rating == -1:
            ratings.append([1, 0, 0])
        elif rating == 0:
            ratings.append([0, 1 ,0])
        elif rating == 1:
            ratings.append([0, 0, 1])
    #d = ['Hi bob builder I am thinking', 'THANKS OBUMMER!!!!!']
    #t = [0, -1]
    a, b = preprocess.full_preprocess(tweets, ratings, 
            [preprocess.tokenize, preprocess.stem_all, preprocess.casing, preprocess.stops, preprocess.punctuation], .2)
    print(len(a[0][0]))
    print(len(b[0][0]))
    #print(a[0])
    #print(a[1])
    print("Basic multi-layer-perceptron")
    #Ternary
    print("\n"+str(tern_basic(a[0], a[1], b[0], b[1], 24, 150)))
    #Binary
    #print("\n"+str(basic_nn(a[0], a[1], b[0], b[1], 24, 150)))
    print("1d Conv nn")
    #Ternary style
    print("\n"+str(tern_convNN(a[0], a[1], b[0], b[1], 24, 40)))
    #Binary style
    #print("\n"+str(convNN(a[0], a[1], b[0], b[1], 24, 10)))
    print("LSTM")
    #Ternary
    print("\n"+str(tern_lstm(a[0], a[1], b[0], b[1], 24, 40)))
    #Binary
    #print("\n"+str(lstm(a[0], a[1], b[0], b[1], 24, 10)))



if __name__ == '__main__':
    main()
