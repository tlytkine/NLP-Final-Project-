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
import networkx as nx 
from textblob import TextBlob 
from nltk import bigrams
import networkx 

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


# # Search twitter for Tweets 

# # Define the search term and the data_since data as variables 
# search_words = "#wildfires"
# date_since = "2018-11-16"

# # To Keep or Remove Retweets 

# # Ignore all retweets 
# new_search = search_words + " -filter:retweets"


# # Get tweets related to climate 
search_term = "#washington+nationals -filter:retweets"

tweets = tw.Cursor(api.search,
 					q=search_term,
 					lang="en",
 					since='2019-11-01').items(300)

all_tweets = [tweet.text for tweet in tweets]


# Remove URLs (links)
def remove_url(txt):

	return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())



tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]



# Analyze sentiment in tweets 

# Create textblob of the tweets 
sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]

# sentiment_objects[0].polarity, sentiment_objects[0]


# Create list of polarity values and tweet text 
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

# print(sentiment_values[0])

# Create dataframe containing the polarity value and tweet text 
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity","tweet"])

# print(sentiment_df.head())

# Plot polarity value in histogram to help highlight overall sentiment toward the subject 

# fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
# sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
#             ax=ax,
#             color="purple")

# plt.title("Sentiments from Tweets on Climate Change")
# plt.show()

# Get and Analyze Tweets Related to the Camp Fire 
search_term = "#washingtonnationals -filter:retweets"

tweets = tw.Cursor(api.search,q=search_term,lang="en",since='2018-09-23').items(200)




# Remove URLs and create textblob object for each tweet
all_tweets_no_urls = [TextBlob(remove_url(tweet.text)) for tweet in tweets]

words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Remove stop words 
tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw]

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

G.add_node("dc",weight=100)

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

# plt.show()

# print(all_tweets_no_urls[:5])

# Calculate polarity of tweets 
wild_sent_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in all_tweets_no_urls]

# Create dataframe containing polarity values and tweet text 
wild_sent_df = pd.DataFrame(wild_sent_values, columns=["polarity","tweet"])
wild_sent_df = wild_sent_df[wild_sent_df.polarity != 0]

print(wild_sent_df.head())







