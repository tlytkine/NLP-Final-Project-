# Import libraries 
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize, sent_tokenize 
import pandas as pd 
import re 
from nltk.corpus import stopwords 
import warnings 
pd.set_option("display.max_colwidth",200)
warnings.filterwarnings("ignore")
import requests 
import csv
import matplotlib.pyplot as plt
from collections import Counter
from heapq import nlargest 
import string 
from string import digits


# 1. Read datasets 
trumpData = pd.read_csv('trumpImpeachment.csv',engine='python',header='infer',names=['Tweet','id','date','source','likes','retweets','sentiment'])

# 2. Clean up data and sort by 100 most common retweets 
trumpData.drop_duplicates(subset=['Tweet'],inplace=True) 
trumpData.dropna(axis=0,inplace=True) 
trumpData["retweets"] = pd.to_numeric(trumpData["retweets"],errors='coerce')
trumpData = trumpData.nlargest(100, columns=['retweets'])

# 4. Get each tweet in a body of text 
trump_tweet_text = []
counter = 0
limit = 100

for tweet in trumpData['Tweet']:
	if(counter==limit):
 		break
	tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', tweet)
	tweet = re.sub('\W+',' ', tweet)
	remove_digits = str.maketrans('', '', digits)
	tweet = tweet.translate(remove_digits)
	# shortword = re.compile(r'\W*\b\w{1,3}\b')
	# shortword.sub('',t)
	# words = set(nltk.corpus.words.words())
	# t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
	trump_tweet_text.append(tweet)
	counter += 1


# 3. Get each tweet in a list 
trump_tweet_list = []
counter = 0
limit = 100 
for tweet in trumpData['Tweet']:
	if(counter==limit):
		break
	tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', tweet)
	tweet = re.sub('\W+',' ', tweet)
	remove_digits = str.maketrans('', '', digits)
	tweet = tweet.translate(remove_digits)
	# shortword = re.compile(r'\W*\b\w{1,3}\b')
	# shortword.sub('',t)
	# words = set(nltk.corpus.words.words())
	# t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
	tweet = ' '.join(tweet.split())
	toAdd = ' '.join( [w for w in tweet.split() if len(w) > 1])
	toAdd.lstrip()
	toAdd.rstrip() 
	trump_tweet_list.append(toAdd.lower())
	counter += 1

# Get eacn tweetID in a list 
trump_tweetID_list = [] 
counter = 0
limit = 100 
for tweetID in trumpData['id']:
	if(counter==limit):
		break
	trump_tweetID_list.append(tweetID)
	counter += 1


tweet_with_id_list = list(zip(trump_tweetID_list, trump_tweet_list))
tweet_with_id_df = pd.DataFrame(tweet_with_id_list, columns=['TweetID','Tweet'])

# 5. Create bigram frequency table for tweet id based on bigrams
stopwords = nltk.corpus.stopwords.words('english')
stopWords = set(stopwords)
bigram_freq_dict = dict() 
bigram_tweetid_dict = dict()
bigram_tweetid_list = []
bigram_freq_list = []
forward_list = [] 
backward_list = [] 



# Generate bigrams for each tweet in each order
for i in range(len(trump_tweet_list)):
	if(i==len(trump_tweet_list)):
		break
	tweet_words = word_tokenize(trump_tweet_list[i])
	tweetID = trump_tweetID_list[i]
	for j in range(len(tweet_words)):
		if(j+1==len(tweet_words)):
			break
		wordA = tweet_words[j]
		wordB = tweet_words[j+1]
		forwardVar = wordA + " " + wordB
		backwardVar = wordB + " " + wordA
		forwardVar1 = str(wordA + " " + wordB + " " + str(tweetID))
		backwardVar1 = str(wordB + " " + wordA + " " + str(tweetID))
		if (forwardVar in bigram_freq_dict or backwardVar in bigram_freq_dict):
			if(forwardVar1 in bigram_tweetid_dict or backwardVar1 in bigram_tweetid_dict):
				bigram_freq_dict[forwardVar] += 1
				bigram_freq_dict[backwardVar] += 1
				bigram_tweetid_dict[forwardVar1] += 1
				bigram_tweetid_dict[backwardVar1] += 1
				forward_list.append(forwardVar)
				backward_list.append(backwardVar)
				bigram_tweetid_list.append(tweetID)
		else:
			bigram_freq_dict[forwardVar] = 1
			bigram_freq_dict[backwardVar] = 1
			bigram_tweetid_dict[forwardVar1] = 1
			bigram_tweetid_dict[backwardVar1] = 1
			forward_list.append(forwardVar)
			backward_list.append(backwardVar)
			bigram_tweetid_list.append(tweetID)



bft = bigram_freq_dict
btd = bigram_tweetid_dict 
# Print results of bigram freq table
# for x in bft:
# 	print("Frequency: %d, Bigram: %s" % (bft[x],x))

# for x in btd:
# 	print("Frequency: %d, TweetID: %s" % (btd[x],x))

# Go through a tweet and look through the bigrams 
# Dictionary with highest frequency tweet for each bigram 
# To the summary, add the tweet for the highest frequency tweets present for each bigram

max_freq = 0
max_tweetid = ""
current_freq = 0
for i in range(len(trump_tweet_list)):
	if(i==len(trump_tweet_list)):
		break
	tweet1 = trump_tweet_list[i]
	tweet_words = word_tokenize(tweet1)
	for j in range(len(tweet_words)):
		for k in range(len(tweet_words)-1):
			wordA = tweet_words[j]
			wordB = tweet_words[k]
			bigram_freq_str = wordA + " " + wordB
			for tweetid in bigram_tweetid_list:
				bigram_freq_str = bigram_freq_str + " " + tweetid
				if(bigram_freq_str in bigram_tweetid_dict):
					freqToAdd = bigram_tweetid_dict[bigram_freq_str]
					current_freq += 1
					if(current_freq > max_freq):
						max_freq = current_freq
						max_tweetid = tweetid
	print("Original Tweet: ")
	print(tweet1)
	print("Summary: ")
	dfVar = tweet_with_id_df.loc[tweet_with_id_df['TweetID'] == max_tweetid]
	print(dfVar['Tweet'])



















