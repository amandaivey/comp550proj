'''
Model setups
'''
import retrieve_data
from sklearn import svm
import preprocess
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD

class lstm_sentiment(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(lstm_sentiment, self).__init__()

        #We will probably need to change this to get it to work correctly with w2v
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1,1,self.hidden_dim)),
                Variable(torch.zeros(1,1,self.hidden_dim)))

    def forward(self, sent):
        embeds = self.word_embeddings(sent)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sent), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def init_baseline(data, targets):
    baseline = svm.SVC()
    baseline.fit(data, targets)
    return baseline

def train_lstm(data, targets, loss_function, input_len, vocab_n, tag_n, epoches):
    model = lstm_sentiment(input_len, tag_n, vocab_n, tag_n)
    optimizer = optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(epoches):
        for sentence, tag in zip(data, targets):
            model.zero_grad()

            model.hidden = model.init_hidden()

            tag_score = model(sentence)

            loss = loss_function(tag_scores, tag)
            loss.backward()
            optimizer.step()
    return model


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
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])
    # fit model
    trainTags=np.asarray(trainTags)
    model.fit(padded, trainTags, epochs=epoc, verbose=0)
    # prepare test data
    testWord_to_ix = word2ix(testSamples)
    encoded_test = [encodeSample(s, testWord_to_ix) for s in testSamples]
    padded_test = pad_sequences(encoded_test, maxlen = maximum_length, padding = 'post')
    # test
    score = model.evaluate(padded_test, testTags, verbose=1)
    return score

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
    
    model.fit(padded_s, trainTags, batch_size=16, epochs=epoc)
    score = model.evaluate(padded_t, testTags, batch_size=16)
    return score


def main():
    #tweet_dict = retrieve_data.open_csv()
    #rating_dict = retrieve_data.get_rating('2download/gold/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
    #for id in tweet_dict.keys():
    #    if id in rating_dict:
    #        print("in it fam")
    alldata = retrieve_data.genLists('binary-diffsep-mydata.csv')
    tweets = alldata[0]
    ratings = alldata[1]
    #d = ['Hi bob builder I am thinking', 'THANKS OBUMMER!!!!!']
    #t = [0, -1]
    a, b = preprocess.full_preprocess(tweets, ratings, 
            [preprocess.tokenize, preprocess.casing, preprocess.stops,
                preprocess.punctuation, preprocess.stem_all], .2)
    print(len(a[0][0]))
    print(len(b[0][0]))
    print("\n"+str(basic_nn(a[0], a[1], b[0], b[1], 24, 300)))
    #print("\n"+str(convNN(a[0], a[1], b[0], b[1], 24, 300)))
   # model1 = Sequential([
   ##         Dense(32, input_shape = (784,)),
   #         Activation('relu'),
   #         Dense(10),
   #         Activation('softmax'),
   #     ])
    # For a single-input model with 2 classes (binary classification):

   # model = Sequential()
   # model.add(Dense(32, activation='relu', input_dim=100))
   # model.add(Dense(1, activation='sigmoid'))
   # model.compile(optimizer='rmsprop',
   #                 loss='binary_crossentropy',
   #                 metrics=['accuracy'])

    # Generate dummy data
    #data = np.random.random((1000, 100))
    #labels = np.random.randint(2, size=(1000, 1))

    # Train the model, iterating on the data in batches of 32 samples
    #model.fit(data, labels, epochs=10, batch_size=32)
    #print(a[0][0])
    #lstm = nn.LSTM(len(a[0][0]), 3)
    #torched = autograd.Variable(torch.Tensor(a[0][0]))
    #hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1,1,3))))
    #out, hidden = lstm(torched.view(1, 1, -1), hidden)
    #train_lstm(a[0], a[1], nn.NLLLoss(), XXX, , 3, 300)
    

if __name__ == '__main__':
    main()
