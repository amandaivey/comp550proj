import csv
import string
import nltk
from nltk import TweetTokenizer

file_with_ratings = "/Users/amandaivey/PycharmProjects/comp550proj/2download/gold/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.txt"

start = "<start>"
end = "<end>"
pad = "<pad>"

def get_rating(file_w_ratings):
    rating_dict = {}
    with open(file_with_ratings) as csvfile:
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
            no_punc = tweet.translate(None, string.punctuation)
            tweet_dict[tweet_id] = no_punc.lower()
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



def main():

    #reads the csv file and puts the tweets into a dictionary of tweet_id: tweet
    tweet_dict = open_csv()
    rating_dict = get_rating(file_with_ratings)

    tweets = generate_lists_for_training(tweet_dict, rating_dict)[0]
    ratings = generate_lists_for_training(tweet_dict, rating_dict)[1]

    print tweets
    print ratings




if __name__ == '__main__':
    main()