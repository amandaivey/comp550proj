from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import sys
import string
import nltk
import csv
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import string
from gensim.models import Word2Vec
import gensim
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

consumer_token = "7QclmKZ0uYE1guPPbmqjZ6i8v"
consumer_secret = "ijFGLclKhx2eOkXrmq0NH1z5cFSjugt97e0x9kCexCbutoFOoc"
access_token = "31629716-JuJrXypLiNUTHxHw4sZIOzHcBoymJLWQDh3e7Mq4p"
access_secret = "00rtcn0Rwff0FyTbr5xCuvUfplVH8TALeLNktUSqI10ev"

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_secret)
#file_to_read = sys.argv[1]

punct = string.punctuation

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
            no_punc = (word.rstrip(punct))
            tweet = tweet + no_punc + " "
            #no_punc = tweet.translate(string.punctuation)
        corpus.append(tweet)
    return corpus

#generates a corpus vector that will be able to be used to generate feature vectors for each tweet sample
def vectorize_corpus(corpus, tweet_dict):
    vectorizer = CountVectorizer()
    #print corpus
    X = vectorizer.fit_transform(corpus)
    return X

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
            V_data.append(model.wv[word])
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
    #print corpus
    vectorize_corpus(corpus, tokenized_tweets)

    corpus_for_w2v = setup_corpus_for_w2v(tokenized_tweets)

    model = gensim.models.Word2Vec(corpus_for_w2v, min_count=1,size=100,workers=4)

    print "Similarity Between Man and Woman: "
    print(model.similarity('man', 'woman'))
    word = ["man", "woman"]
    V_data = model.wv[word]
    V = torch.Tensor(V_data)
    
    tensorize_samples(tokenized_tweets, model)
    print "Three most common words in corpus"
    print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
    print punct



if __name__ == '__main__':
    main()