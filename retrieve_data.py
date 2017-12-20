import csv
import string
import nltk
from nltk import TweetTokenizer
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy

consumer_token = "7QclmKZ0uYE1guPPbmqjZ6i8v"
consumer_secret = "ijFGLclKhx2eOkXrmq0NH1z5cFSjugt97e0x9kCexCbutoFOoc"
access_token = "31629716-JuJrXypLiNUTHxHw4sZIOzHcBoymJLWQDh3e7Mq4p"
access_secret = "00rtcn0Rwff0FyTbr5xCuvUfplVH8TALeLNktUSqI10ev"

#file_with_ratings = "/Users/amandaivey/PycharmProjects/comp550proj/2download/gold/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.txt"

start = "<start>"
end = "<end>"
pad = "<pad>"

def get_rating(file_w_ratings):
    rating_dict = {}
    with open(file_w_ratings) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tweet = row[0].split()
            tweet_id = tweet[0]
            rating = tweet[1]
            rating_dict[tweet_id] = rating
    return rating_dict

def open_csv():
    tweet_dict = {}
    with open('twitter_data_3pt.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tweet_id = row[0]
            tweet = row[1]
            tweet_dict[tweet_id] = tweet
    return tweet_dict

def tokenize_tweets(tweet_dict):
    tokenized_tweets = {}
    tknzr = TweetTokenizer()
    for k,v in tweet_dict.iteritems():
        tokenized_tweet = tknzr.tokenize(v)
        tokenized_tweets[k] = tokenized_tweet
    return tokenized_tweets

def generate_lists_for_training(tweet_dict, rating_dict):
    tweet_list = []
    rating_list = []
    for k,v in tweet_dict.iteritems():
        tweet_list.append(tweet_dict[k])
        rating_list.append(rating_dict[k])
    return (tweet_list, rating_list)

def get_tweet_length(tweet):
    tokens = tweet.split()
    length = 0
    for word in tokens:
        length += 1
    return length

def pad_tweets(tweets, max_len):
    padded_tweets = []
    for tweet in tweets:
        tokened_tweet = start + " " + tweet
        length = get_tweet_length(tokened_tweet)
        padded_tweet = tokened_tweet
        if length < max_len + 2:
            while length < max_len + 2:
                padded_tweet = padded_tweet + " "  + pad
                #print padded_tweet
                length += 1
            padded_tweet = padded_tweet + " " + end
        else:
            padded_tweet = padded_tweet + " " + end
        padded_tweets.append(padded_tweet)
    return padded_tweets


#gets maximum tweet length based on number of words
def get_max_len(tweet_dict):
    max_len = 0
    for tweet in tweet_dict.itervalues():
        len = 0
        for word in tweet.split():
            len += 1
        if len > max_len:
            max_len = len
    return max_len

def gather_tweets(filepath):
    tweet_to_rating = dict()
    f = open(filepath, 'r')
    for line in f:
        tweet_id, rating = line.split()
        tweet = 
    return tweet_to_rating


def main():

    #reads the csv file and puts the tweets into a dictionary of tweet_id: tweet
    tweet_dict = open_csv()
    #rating_dict = get_rating(file_with_ratings)

    max_tweet_length = get_max_len(tweet_dict)

    #tweets = generate_lists_for_training(tweet_dict, rating_dict)[0]
    #ratings = generate_lists_for_training(tweet_dict, rating_dict)[1]

    #print tweets
    #print ratings


    #padded_tweets = pad_tweets(tweets, max_tweet_length)

    #print padded_tweets


if __name__ == '__main__':
    main()
