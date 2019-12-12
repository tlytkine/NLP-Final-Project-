from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re

### Client ###
class TwitterClient():
    def __init__(self):
        self.auth = TwitterAuthenticator().authenticate_twitter()
        self.twitter_client = API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def get_user_tweets(self, user, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def search_twitter(self, query, num_tweets):
        tweets = []
        query += ' -filter:retweets'            #filter retweets
        for tweet in Cursor(self.twitter_client.search, q=query, tweet_mode='extended').items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_api(self):
        return self.twitter_client

### Authenticator ###
class TwitterAuthenticator():
    def authenticate_twitter(self):
        with open('twitter_credentials.json', 'r') as file:
            creds = json.load(file)

        auth = OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
        auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
        return auth

### Analyzer ###
#tools for performing analysis on tweets
class TweetAnalyzer():
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        # if analysis.sentiment.polarity > 0:
        #     return 1
        # elif analysis.sentiment.polarity == 0:
        #     return 0
        # else:
        #     return -1

        return analysis.sentiment.polarity

    def to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['Tweet'])
        df['id'] = np.array([tweet.id for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        df['sentiment'] = np.array([self.analyze_sentiment(tweet.full_text) for tweet in tweets])
        return df


if __name__ == '__main__':
    client = TwitterClient()
    analyzer = TweetAnalyzer()

    tweets = client.search_twitter('trump impeachment', 10000)
    trump = analyzer.to_data_frame(tweets)
    print(trump.head(10))
    trump.to_csv('trumpImpeachment.csv', index=False)

    tweets = client.search_twitter('washington nationals', 10000)
    nats = analyzer.to_data_frame(tweets)
    print(nats.head(10))
    nats.to_csv('washingtonNationals.csv', index=False)

    tweets = client.search_twitter('washington redskins', 10000)
    skins = analyzer.to_data_frame(tweets)
    print(skins.head(10))
    skins.to_csv('washingtonRedskins.csv', index=False)

    tweets = client.search_twitter('dc metro -area', 10000)
    metro = analyzer.to_data_frame(tweets)
    print(metro.head(10))
    metro.to_csv('dcMetro.csv', index=False)
