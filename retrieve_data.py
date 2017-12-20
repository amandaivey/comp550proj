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

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_secret)

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

def gather_tweets(ratingTranslate, filepath):
    tweet_to_rating = dict()
    api = tweepy.API(auth, wait_on_rate_limit=True)
    f = open(filepath, 'r')
    errors = 0
    for line in f:
        try:
            tweet_id, rating = line.split()
            tweet = api.get_status(tweet_id).text
            tweet_to_rating[tweet] = ratingTranslate[rating]
            print(tweet)
            print(rating)
            print("**************")
        except tweepy.TweepError, e:
            pass
            errors+=1
            print(errors)
            print(e)
    f.close()
    return tweet_to_rating

def write_dict_to_csv(dictToWrite, filename):
    f = open(filename, 'w')
    for k, v in dictToWrite.items():
        f.write(k+","+v+"\n")
    print("File written")


def assembleBatches(readfile):
    fr = open(readfile, "r")
    fullList = []
    ratingLookUp = dict()
    for line in fr:
        tweet_id, rating = line.split()
        fullList.append((tweet_id, rating))
        ratingLookUp[tweet_id] = rating
    chunks = [fullList[x:x+100] for x in xrange(0, len(fullList), 100)]
    fr.close()
    return (chunks, ratingLookUp)

def batch_grab(writefile, batches, ratinglookup, translateRating):
    fw = open(writefile, "w")
    api = tweepy.API(auth, wait_on_rate_limit=True)
    count = 0
    for batch in batches:
        ids, ratings = zip(*batch)
        idsToWrite = api.statuses_lookup(ids)
        print(len(idsToWrite))
        for obj in idsToWrite:
            if obj.text == None:# or obj.id not in ratinglookup:
                print('dead tweet encountered')
                pass
            else:
                text = [char for char in obj.text if char != "|" and char != "\n"]
                fw.write(''.join(text)+"|"+str(translateRating[ratinglookup[obj.id_str]])+"\n")
        print(count)
        count+=1
    fw.close()

def main():

    #reads the csv file and puts the tweets into a dictionary of tweet_id: tweet
    tweet_dict = open_csv()
    #rating_dict = get_rating(file_with_ratings)

    max_tweet_length = get_max_len(tweet_dict)

    #tweets = generate_lists_for_training(tweet_dict, rating_dict)[0]
    #ratings = generate_lists_for_training(tweet_dict, rating_dict)[1]

    #print tweets
    #print ratings

    ratingT = {'positive':1, 'negative':-1, 'neutral':0}

    #ttr = gather_tweets(ratingT, '2download/gold/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
    #write_dict_to_csv(ttr, "myratings.csv")

    # Batch version
    # Make a list of lists
    # inner lists consist of 100 tuples each consisting of the form (id, rating)
    # Then fetch ids using batch lookup which fetches 100 at a time, setting 'map' to true to return None objects for errors
    # Continuously write each one to a csv as they are grabbed
    batches, ratinglook = assembleBatches('2download/gold/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.txt')
    batch_grab('diffsep-mydata.csv', batches, ratinglook, ratingT)


    #padded_tweets = pad_tweets(tweets, max_tweet_length)

    #print padded_tweets


if __name__ == '__main__':
    main()
