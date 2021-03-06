from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import sys
import string
import nltk
import csv
from nltk.tokenize import TweetTokenizer
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy
from gensim.models import Word2Vec
import gensim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import svm

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

'''
Basic neural network class for a classifier on tweets.
Requires as input the number of possible tags and the number of possible tokens.
Source: Tutorial on Deep Learning for NLP on the PyTorch site.
'''
class tweetClassifier(nn.Module):
    def __init__(self, n_tags, n_tokens):
        super(tweetClassifier, self).__init__()
        # Parameters
        self.linear = nn.Linear(n_tokens, n_tags)

    def forward(self, tweet_vec):
        return F.log_softmax(self.linear(tweet_vec))

# NN Functions, similarly from the tutorial
def make_tweet_vect(tokenized_tweet, token_to_index):
    vect = torch.zeros(len(token_to_index))
    for token in tokenized_tweet:
        vect[token_to_index[token]] += 1
    return vect.view(1, -1)

def make_target(tag, tag_to_index):
    return torch.LongTensor([tag_to_index[tag]])

'''
Given training and test data in the format of a list of tuples of the form (tokenized tweet, tag)
Returns a dictionary mapping from any given token to the index of its type in the tweet vector.
'''
def build_token_indices(training_data, test_data):
    indices = {}
    for tweet, _ in training_data+test_data:
        for token in tweet:
            if token not in indices:
                indices[token] = len(indices)
    return indices

'''
Takes as input the following:
    training_data: A list of data in the form of (tokenized tweet, rating).
    epoches: The number of epoches for which this model will be trained.
    token_indices: A dictionary containing all possible types in the dataset, which allows one to map 
        from a token to the location of its type in the tweet vector.
    tag_indices: A dictionary mapping from any given tag in the dataset to its index in an output vector.
    loss_function: The loss function used for training.
    learning_rate: The learning rate used to initialize the model.
Returns a tweetClassifer model trained to these specifications.
'''
def train_tc(training_data, epoches, token_indices, tag_indices, loss_function, learning_rate):
    model = tweetClassifier(len(tag_indices), len(token_indices))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epoches):
        for tweet, tag in training_data:
            # Zero the gradients
            model.zero_grad()

            # Assemble the vector representation of the tweet and the target
            tweet_vect = Variable(make_tweet_vect(tweet, token_indices))
            target = Variable(make_target(tag, tag_indices))

            # Forward pass
            log_probs = model(tweet_vect)

            # Loss, Grad, update with optimizer
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
    return model

def test(model, test_data, token_indices):
    for tweet, tag in test_data:
        tweet_vect = Variable(make_tweet_vect(tweet, token_indices))
        log_probs = model(tweet_vect)
        print(log_probs)

def run_example_data():
    sample_train = [(['I', 'hate', 'pie'],"n"), (['I', 'love', 'cats'],"p"), (['Cake', 'is', 'alright', ':)'],"p")]

    sample_test = [(['I', 'love', 'cake'], "p")]

    tag_to_index = {"n": 0,  "p": 1}
    tok_indices = build_token_indices(sample_train, sample_test)

    model = train_tc(sample_train, 100, tok_indices, tag_to_index, nn.NLLLoss(), 0.1)
    test(model, sample_test, tok_indices)


#generates a corpus, a list of lists the inner list is a tokenized tweet representing a single sample
#returns this list to be vectorized
def generate_corpus(tweet_dict):
    corpus = []
    for k,v in tweet_dict.iteritems():
        tweet = ""
        for word in v:
            no_punc = (word.rstrip(string.punctuation))
            tweet = tweet + no_punc + " "
            #no_punc = tweet.translate(string.punctuation)
        corpus.append(tweet)
    return corpus

#returns a list of lists where the outer list is a sample containing a list of tokens in each sample
def setup_corpus_for_w2v(tokenized_tweets):
    sample_list = []
    for tweet in tokenized_tweets.itervalues():
        sample_list.append(tweet)
    return sample_list

#takes the tokenized sentences and word2vec model and forms tensors for each sentence
#will require padding
def tensorize_samples(tokenized_tweets, model):
    tensors = []
    max_length = 0
    for tweet in tokenized_tweets.itervalues():
        V_data = []
        for word in tweet:
            V_data.append(torch.Tensor(model.wv[word]))
        tensors.append(V_data)
    return tensors

def init_baseline(id_to_data, tweet_ratings, rating_to_val):
    data = []
    labels = []
    for k, v in id_to_data.items():
        data.append(v)
        labels.append(rating_to_val[tweet_ratings[k]])
    baseline_clf = svm.SVC()
    baseline_clf.fit(data, labels)
    return baseline_clf


def main():

    #tokenized_tweets is a dictionary of tweet_id : tokenized_tweet
    tokenized_tweets = tokenize_tweets(tweet_dict)
    #print tokenized_tweets

    corpus = generate_corpus(tokenized_tweets)

    corpus_for_w2v = setup_corpus_for_w2v(tokenized_tweets)

    #messing word 2 vec
    model = gensim.models.Word2Vec(corpus_for_w2v, min_count=1,size=100,workers=4)
    print("Similarity Between Man and Woman: ")
    print(model.similarity('man', 'woman'))
    word = ["man", "woman"]
    V_data = model.wv[word]
    print "Three most common words in corpus"
    print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
    
    #creates an individual tensor of an individual word
    #V = torch.Tensor(V_data)

    #creates tensors for each tweet, these need to be padded
    tensors = tensorize_samples(tokenized_tweets, model)
    #print tensors

    #lab_to_val = {"positive": 1, "negative": -1, "neutral": 0}
    
    #Creates a baseline classifier using svc
    #baseline_classifier = init_baseline(Dictionary from id to word embedding, 
                                        #dict from id to rating, dict mapping label to value)

    #Messing around with the nn.LSTM module
    #nn.LSTM(INPUT LAYER SIZE, HIDDEN LAYER SIZE)
    #lstm = nn.LSTM(100, 4)
    #hidden = (Variable(torch.randn(1,1,4)),
    #          Variable(torch.randn((1,1,4))))
    #for data in tensors:
    #    for t in data:
    #        v = Variable(t)
    #        out, hidden = lstm(v.view(1,1,-1), hidden)
    #print(out)
    #print(hidden)



if __name__ == '__main__':
    main()
