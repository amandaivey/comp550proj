from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import sys
import nltk
import csv
from nltk.tokenize import TweetTokenizer


consumer_token = "7QclmKZ0uYE1guPPbmqjZ6i8v"
consumer_secret = "ijFGLclKhx2eOkXrmq0NH1z5cFSjugt97e0x9kCexCbutoFOoc"
access_token = "31629716-JuJrXypLiNUTHxHw4sZIOzHcBoymJLWQDh3e7Mq4p"
access_secret = "00rtcn0Rwff0FyTbr5xCuvUfplVH8TALeLNktUSqI10ev"

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_secret)
file_to_read = sys.argv[1]

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
            tweet_dict[tweet_id] = tweet
    return tweet_dict


def main():

    # Construct the API instance
    api = tweepy.API(auth)

    #get tweet based on id
    #tweet = api.get_status(629186282179153920)
    #print(tweet)

    #dictionary of ids to rating so that you can find the rating of a given tweet
    id_dict = read_file(file_to_read)
    print id_dict

    #reads the csv file and puts the tweets into a dictionary of tweet_id: tweet
    tweet_dict = open_csv()
    print tweet_dict

    #tokenized_tweets is a dictionary of tweet_id : tokenized_tweet
    tokenized_tweets = tokenize_tweets(tweet_dict)
    print tokenized_tweets

    #write_to_csv(tweet_dict, '/Users/amandaivey/PycharmProjects/comp550proj/twitter_data.csv')


if __name__ == '__main__':
    main()
