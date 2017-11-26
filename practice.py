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

#reads a tab delimited text file of the format tweet id, topic, rating
#and puts this info into a python dict of the form tweetid: (topic, rating)
#returns dict
def read_file(file_name):
    id_dict = {}
    with open(file_name, 'r+') as f:
        content = f.readlines()
    for item in content:
        id_dict[item.strip().split("\t")[0]] = (item.strip().split("\t")[1], item.strip().split("\t")[2])
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
            writer.writerow([key, value])

def tokenize_tweets(tweet_dict):
    tknzr = TweetTokenizer()

def main():

    # Construct the API instance
    api = tweepy.API(auth)

    #get tweet based on id
    #tweet = api.get_status(629186282179153920)
    #print(tweet)

    id_dict = read_file(file_to_read)
    print id_dict
    tweet_dict = get_tweets_from_id(id_dict)
    print tweet_dict

    write_to_csv(tweet_dict, '/Users/amandaivey/PycharmProjects/comp550proj/twitter_data.csv')


if __name__ == '__main__':
    main()