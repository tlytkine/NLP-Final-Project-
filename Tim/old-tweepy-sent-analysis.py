# Apply sentiment analysis to Twitter data using Python package textblob
# Calculate polarity value for each tweet on a given subject and plot 
# these values in a histogram to identify the overall sentiment toward 
# the subject of interest 

# Get and clean tweets related to climate 
# Begin by reviewing how to search for and clean tweets that you will use to
# analyze sentiments in Twietter data 

# test - force trigger rebuild 

import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import itertools 
import collections 

import tweepy as tw 
import nltk 
from nltk.corpus import stopwords 
import re 
import networkx 
from textblob import TextBlob 
from nltk import bigrams

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# Define keys 
consumer_key = 'vsWgnpHRQAbdq1tmzSR98brBm'
consumer_secret = 'mC3VDP10dskEVwg9SxMpapj290XSX7oG1sEFPokSm18SZExE03'
access_token = '1196470873668108289-5on7gC1NCRly2fTHMDC5Mh7OEyVnvg'
access_token_secret = 'PjIIHwBxOtR3lYL5l1ERxqs5oN5jnvqflKNtK7YqvwK6O'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit = True)

# 18,000 is max number of tweets you can request from twitter at 
# one time 

# Structure of a Tweet: 
# User Name, Time Stamp, Tweet Text, Hashtags, Links, Embedded Media,
# Replies, Retweets, Favorites, Latitude / Longitude 

# Twitter REST API only allows you to access tweets from the last 6-9 days
# python-twitter or tweepy can be used 

# Text Mining and Cleaning is done with nltk and re libraries 

# Send a Tweet using API Access
# Post a tweet from Python
# api.update_status("This is a test of tweeting from Python.")


# Search twitter for Tweets 

# Define the search term and the data_since data as variables 
search_words = "#wildfires"
date_since = "2018-11-16"

# # Collect tweets 
# tweets = tw.Cursor(api.search,
# 				q = search_words,
# 				lang="en",
# 				since=date_since).items(5) #5 most recent tweets 
# print(tweets)

# .Cursor() returns an object that you can iterate or loop over to access the data collected 
# Each item in the iterator has various attributes that you can access to get information
# about each tweet including: 
# 1. the text of the tweet 
# 2. who sent the tweet 
# 3. the data the tweet was sent... and more 
"""
tweetcount = 0 
# Iterate and print tweets 
for tweet in tweets:
	tweetcount = tweetcount + 1 
	print("Tweet %d " % (tweetcount))
	print(tweet.text)
"""
# Collect a list of tweets 
# tweetlist = [tweet.text for tweet in tweets]
# tweetcount = 0 
# for i in range(len(tweetlist)):
# 	tweetcount = tweetcount + 1 
#	print("Tweet %d " % (tweetcount))
#	print(tweetlist[i])

# To Keep or Remove Retweets 

# Ignore all retweets 
new_search = search_words + " -filter:retweets"

# tweets = tw.Cursor(api.search,
#                        q=new_search,
#                        lang="en",
#                        since=date_since).items(5)

# tweetlist = [tweet.text for tweet in tweets]


# tweet.user.screen_name provides the user's twitter handle associated with each tweet
# tweet.user.location provides the user's provided location. 

# tweets = tw.Cursor(api.search,
# 					q=new_search,
# 					lang="en",
# 					since=date_since).items(1000)

# user_locs =[[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
# user_locs 

# for screen_name, location in user_locs:
# 	if(location == 'Washington, DC'):
# 		print("User Name: %s" % (screen_name))
# 		print("User Location: %s" % (location))

"""
tweetcount = 0 
for i in range(len(tweetlist)):
	tweetcount = tweetcount + 1 
	print("Tweet %d " % (tweetcount))
	print(tweetlist[i])
"""

# # Create a Pandas Dataframe from a list of Tweet Data
# tweet_text = pd.DataFrame(data=user_locs,columns=['user','location'])

# # print(tweet_text)

# new_search = "climate+change -filter:retweets"

# tweets = tw.Cursor(api.search,q=new_search,lang="en",since='2018-04-23').items(1000)

# all_tweets = [tweet.text for tweet in tweets]
# print(all_tweets[:5])

# Lesson 3: Analyze Word Frequency Counts Using Twitter Data and Tweepy in Python
# How to:
# 1. Remove URL's from tweets 
# 2. Clean up tweet text, including differences in case (e.g. upper, lower) that will affect
# unique word counts and removing words that are not useful for the analysis 
# 3. Summarize and count words found in tweets 


# Get tweets related to climate 
search_term = "#climate+change -filter:retweets"

tweets = tw.Cursor(api.search,
					q=search_term,
					lang="en",
					since='2018-11-01').items(100)

all_tweets = [tweet.text for tweet in tweets]


# Remove URLs (links)
def remove_url(txt):
	"""Replace URLs found in a text string with nothing 
	(i.e. it will remove the URL from the string).

	Parameters 
	-----------
	txt : string 
		A text string that you want to parse and remove urls 

	Returns 
	----------
	The same txt string with url's removed 
	"""

	return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
# print(all_tweets_no_urls[:5])

# Text Cleanup - Address Case Issues 

# print(all_tweets_no_urls[0].lower().split())
# Create a list of lists containing lowercase words for each tweet 
words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
# print((words_in_tweet[:2])

# List of all words across tweets
# Python library collections helps create special type of Python dictionary
# to get count of how many times each word appears in the sample 
# Has built in method most_common that returns the most commonly used words
# and the number of times that they are used 

# List of all words across tweets 
# all_words_no_urls = list(itertools.chain(*words_in_tweet))

# Create counter
# counts_no_urls = collections.Counter(all_words_no_urls)

# print(counts_no_urls.most_common(15))
# Create Pandas Dataframe for analysis and plotting that includes only the 
# top 15 most common words

# clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
# 							columns=['words','count'])

# print(clean_tweets_no_urls.head())


# fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
# clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
# 									y='count',
# 									ax=ax,
# 									color="blue")

# ax.set_title("Common Words Found in Tweets (Including All Words")

# plt.show()

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Remove stop words 
tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

# print(tweets_nsw[0])

# all_words_nsw = list(itertools.chain(*tweets_nsw))

# counts_nsw = collections.Counter(all_words_nsw)

# # print(counts_nsw.most_common(15))

# clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(15), columns=['words','count'])

# fig, ax = plt.subplots(figsize=(10,10))

# clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
# 							y='count',
# 							ax=ax,
# 							color="blue")

# ax.set_title("Common Words Found in Tweets (Without Stop Words)")

# plt.show()

# Remove collection words 
collection_words = ['climatechange', 'climate', 'change']
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]

# Exploring co-occuring words (Bigrams)

# Create a list of lits containing bigrams in tweets 
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

# View bigrams for the first tweet 
print(terms_bigram[0])



# # Flatten list of words in clean tweets
# all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))

# # Create counter of words in clean tweets
# counts_nsw_nc = collections.Counter(all_words_nsw_nc)


# clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(100),
#                              columns=['words', 'count'])
# clean_tweets_ncw.head()



# fig, ax = plt.subplots(figsize=(8, 8))

# # Plot horizontal bar graph
# clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',
#                       y='count',
#                       ax=ax,
#                       color="purple")

# ax.set_title("Common Words Found in Tweets (Without Stop or Collection Words)")

# plt.show()


# Apply sentiment analysis to Twitter data using Python package textblob
# Calculate polarity value for each tweet on a given subject and plot 
# these values in a histogram to identify the overall sentiment toward 
# the subject of interest 

# Get and clean tweets related to climate 
# Begin by reviewing how to search for and clean tweets that you will use to
# analyze sentiments in Twietter data 

# test - force trigger rebuild 

import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import itertools 
import collections 

import tweepy as tw 
import nltk 
from nltk.corpus import stopwords 
import re 
import networkx 
from textblob import TextBlob 
from nltk import bigrams
import networkx as nx 

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# Define keys 
consumer_key = 'vsWgnpHRQAbdq1tmzSR98brBm'
consumer_secret = 'mC3VDP10dskEVwg9SxMpapj290XSX7oG1sEFPokSm18SZExE03'
access_token = '1196470873668108289-5on7gC1NCRly2fTHMDC5Mh7OEyVnvg'
access_token_secret = 'PjIIHwBxOtR3lYL5l1ERxqs5oN5jnvqflKNtK7YqvwK6O'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit = True)


# Search twitter for Tweets 

# Define the search term and the data_since data as variables 
search_words = "#wildfires"
date_since = "2018-11-16"

# To Keep or Remove Retweets 

# Ignore all retweets 
new_search = search_words + " -filter:retweets"


# Get tweets related to climate 
search_term = "#climate+change -filter:retweets"

tweets = tw.Cursor(api.search,
					q=search_term,
					lang="en",
					since='2018-11-01').items(100)

all_tweets = [tweet.text for tweet in tweets]


# Remove URLs (links)
def remove_url(txt):


	return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]

# Text Cleanup - Address Case Issues 


words_in_tweet = [tweet.lower().split() for tweet in tweets_no_urls]

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Remove stop words 
tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]


# Remove collection words 
collection_words = ['climatechange', 'climate', 'change']
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]

# Exploring co-occuring words (Bigrams)

# Create a list of lits containing bigrams in tweets 
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

# View bigrams for the first tweet 
# print(terms_bigram[0])

# Original tweet without URLs
# print(tweets_no_urls[0])

# Clean tweet 
# print(tweets_nsw_nc[0])

# Similar to what you learned in the previous lesson on word frequency counts,
# you can use a counter to capture the bigrams as dictionary kets and their counts 
# as dictionary values.

# Begin by flattening the list of bigrams. You can then create the counter and 
# query the top 20 most common bigrams across the tweets 

# Flatten list of bigrams in clean tweets 
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams 
bigram_counts = collections.Counter(bigrams)

print(bigram_counts.most_common(20))

bigram_df = pd.DataFrame(bigram_counts.most_common(20),
							columns=['bigram','count'])

# print(bigram_df)

# Visualize Networks of Bigrams 

# Create dictionary of bigrams and their counts 
d = bigram_df.set_index('bigram').T.to_dict('records')

# Create network plot 
G = nx.Graph() 

# Create connections between nodes 
for k, v in d[0].items():
	G.add_edge(k[0], k[1], weight=(v * 10))

G.add_node("china",weight=100)

fig, ax = plt.subplots(figsize=(10, 8))

pos = nx.spring_layout(G, k=1)

# Plot networks 
nx.draw_networkx(G, pos,
					font_size=16,
					width=3,
					edge_color='grey',
					node_color='purple',
					with_labels= False,
					ax=ax)

# Create offset labels 
for key, value in pos.items():
	x, y = value[0]+0.135, value[1]+0.045
	ax.text(x, y,
			s=key,
			bbox=dict(facecolor='red', alpha=0.25),
			horizontalalignment='center',fontsize=13)

plt.show()


# Text Cleanup - Address Case Issues 


# words_in_tweet = [tweet.lower().split() for tweet in tweets_no_urls]

# # nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Remove stop words 
# tweets_nsw = [[word for word in tweet_words if not word in stop_words]
#               for tweet_words in words_in_tweet]


# # Remove collection words 
# collection_words = ['climatechange', 'climate', 'change']
# tweets_nsw_nc = [[w for w in word if not w in collection_words]
#                  for word in tweets_nsw]

# # Exploring co-occuring words (Bigrams)

# # Create a list of lits containing bigrams in tweets 
# terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

# # View bigrams for the first tweet 
# # print(terms_bigram[0])

# # Original tweet without URLs
# # print(tweets_no_urls[0])

# # Clean tweet 
# # print(tweets_nsw_nc[0])

# # Similar to what you learned in the previous lesson on word frequency counts,
# # you can use a counter to capture the bigrams as dictionary kets and their counts 
# # as dictionary values.

# # Begin by flattening the list of bigrams. You can then create the counter and 
# # query the top 20 most common bigrams across the tweets 

# # Flatten list of bigrams in clean tweets 
# bigrams = list(itertools.chain(*terms_bigram))

# # Create counter of words in clean bigrams 
# bigram_counts = collections.Counter(bigrams)

# print(bigram_counts.most_common(20))

# bigram_df = pd.DataFrame(bigram_counts.most_common(20),
# 							columns=['bigram','count'])

# # print(bigram_df)

# # Visualize Networks of Bigrams 

# # Create dictionary of bigrams and their counts 
# d = bigram_df.set_index('bigram').T.to_dict('records')

# # Create network plot 
# G = nx.Graph() 

# # Create connections between nodes 
# for k, v in d[0].items():
# 	G.add_edge(k[0], k[1], weight=(v * 10))

# G.add_node("china",weight=100)

# fig, ax = plt.subplots(figsize=(10, 8))

# pos = nx.spring_layout(G, k=1)

# # Plot networks 
# nx.draw_networkx(G, pos,
# 					font_size=16,
# 					width=3,
# 					edge_color='grey',
# 					node_color='purple',
# 					with_labels= False,
# 					ax=ax)

# # Create offset labels 
# for key, value in pos.items():
# 	x, y = value[0]+0.135, value[1]+0.045
# 	ax.text(x, y,
# 			s=key,
# 			bbox=dict(facecolor='red', alpha=0.25),
# 			horizontalalignment='center',fontsize=13)

# plt.show()









