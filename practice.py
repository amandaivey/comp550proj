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


consumer_token = "7QclmKZ0uYE1guPPbmqjZ6i8v"
consumer_secret = "ijFGLclKhx2eOkXrmq0NH1z5cFSjugt97e0x9kCexCbutoFOoc"
access_token = "31629716-JuJrXypLiNUTHxHw4sZIOzHcBoymJLWQDh3e7Mq4p"
access_secret = "00rtcn0Rwff0FyTbr5xCuvUfplVH8TALeLNktUSqI10ev"

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_secret)
#file_to_read = sys.argv[1]

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
def train(training_data, epoches, token_indices, tag_indices, loss_function, learning_rate):
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

    model = train(sample_train, 100, tok_indices, tag_to_index, nn.NLLLoss(), 0.1)
    test(model, sample_test, tok_indices)

#reads a tab delimited text file of the format tweet id, topic, rating
#and puts this info into a python dict of the form tweetid: rating
#returns dict
def read_file(file_name):
    id_dict = {}
    with open(file_name, 'r+') as f:
        content = f.readlines()
    for item in content:
        id_dict[item.strip().split("\t")[0]] = item.strip().split("\t")[2]
    return id_dict


def get_tweets_from_id(id_dict):
    tweet_dict = {}
    num_e = 0
    api = tweepy.API(auth)
    for k,v in id_dict.iteritems():
        try:
            tweet = api.get_status(k)
            text = tweet.text
            tweet_dict[k] = (text, id_dict[k][1])
        except tweepy.TweepError, e:
            pass
            num_e += 1
            #print num_e
            continue

    return tweet_dict

def write_to_csv(mydict, directory):
    with open(directory, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value[0]])

def tokenize_tweets(tweet_dict):
    tokenized_tweets = {}
    tknzr = TweetTokenizer()
    for k,v in tweet_dict.iteritems():
        tokenized_tweet = tknzr.tokenize(v)
        tokenized_tweets[k] = tokenized_tweet
    return tokenized_tweets

def open_csv():
    tweet_dict = {}
    with open('twitter_data.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tweet_id = row[0]
            tweet = row[1]
            no_punc = tweet.translate(None, string.punctuation)
            tweet_dict[tweet_id] = no_punc.lower()
    return tweet_dict

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


def main():

    # Construct the API instance
    api = tweepy.API(auth)

    #get tweet based on id
    #tweet = api.get_status(629186282179153920)
    #print(tweet)

    #dictionary of ids to rating so that you can find the rating of a given tweet
    #id_dict = read_file(file_to_read)
    #print id_dict

    #reads the csv file and puts the tweets into a dictionary of tweet_id: tweet
    tweet_dict = open_csv()
    #print tweet_dict

    #tokenized_tweets is a dictionary of tweet_id : tokenized_tweet
    tokenized_tweets = tokenize_tweets(tweet_dict)
    #print tokenized_tweets

    #write_to_csv(tweet_dict, '/Users/amandaivey/PycharmProjects/comp550proj/twitter_data.csv')
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

    #Creates a baseline SVM classifier, using SVC
    baseline_classifier = svm.SVC()

    #Messing around with the nn.LSTM module
    #nn.LSTM(INPUT LAYER SIZE, HIDDEN LAYER SIZE)
    lstm = nn.LSTM(100, 4)
    hidden = (Variable(torch.randn(1,1,4)),
              Variable(torch.randn((1,1,4))))
    for data in tensors:
        for t in data:
            v = Variable(t)
            out, hidden = lstm(v.view(1,1,-1), hidden)
    print(out)
    #print(hidden)


if __name__ == '__main__':
    main()
