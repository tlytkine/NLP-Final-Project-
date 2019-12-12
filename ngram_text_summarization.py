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

print("ngram Text Summarization Model")

# 1. Read datasets 
print("Reading in data...")
trumpData = pd.read_csv('trumpImpeachment.csv',engine='python',header='infer',names=['Tweet','id','date','source','likes','retweets','sentiment'])

# 2. Clean up data and sort by 100 most common retweets 
print("Preprocessing: cleaning up and organizing the data...")
trumpData.drop_duplicates(subset=['Tweet'],inplace=True) 
trumpData.dropna(axis=0,inplace=True) 
trumpData["retweets"] = pd.to_numeric(trumpData["retweets"],errors='coerce')
# 70-30 ratio
# Training Set 
trumpData = trumpData.nlargest(140, columns=['retweets'])
# Testing Set 
trumpDataCopy = trumpData.nlargest(60, columns=['retweets'])


# Use 140 to train
# Use 60 to test 
# 4. Get each tweet in a body of text 
# Training data
trump_tweet_text = ""
for tweet in trumpData['Tweet']:
	tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', tweet)
	tweet = re.sub('\W+',' ', tweet)
	remove_digits = str.maketrans('', '', digits)
	tweet = tweet.translate(remove_digits)
	# shortword = re.compile(r'\W*\b\w{1,3}\b')
	# shortword.sub('',t)
	# words = set(nltk.corpus.words.words())
	# t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
	trump_tweet_text += tweet
	trump_tweet_text += "."

# 3. Get each tweet in a list 
trump_tweet_list = []
for tweet in trumpData['Tweet']:
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
	var = toAdd.split() 
	count = 0 
	for x in var:
		if(count==0):
			x.lstrip()
			x.rstrip() 
			x[0].capitalize()
			x = ' '.join(x)
		else:
			x.lstrip()
			x.rstrip()
			x.lower()
		count = count + 1
	toAdd = ' '.join(var)
	trump_tweet_list.append(toAdd + ".")

# Testing data
test_text_trump = ""
for tweet in trumpDataCopy['Tweet']:
	tweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', tweet)
	# tweet = re.sub('\W+',' ', tweet)
	remove_digits = str.maketrans('', '', digits)
	tweet = tweet.translate(remove_digits)
	tweet.lstrip()
	tweet.rstrip()
	tweet = tweet.split()
	tweet[0].capitalize()
	tweet = ' '.join(tweet)
	# shortword = re.compile(r'\W*\b\w{1,3}\b')
	# shortword.sub('',t)
	# words = set(nltk.corpus.words.words())
	# t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
	test_text_trump += tweet
	test_text_trump += "."

# Testing data
test_list_trump = []
for tweet in trumpDataCopy['Tweet']:
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
	test_list_trump.append(toAdd.lower())

# Get eacn tweetID in a list 
print("Getting tweetIDS...")
trump_tweetID_list = [] 
for tweetID in trumpData['id']:
	trump_tweetID_list.append(tweetID)

tweetidtest = trump_tweetID_list[4]

tweet_with_id_list = list(zip(trump_tweetID_list, trump_tweet_list))
tweet_with_id_df = pd.DataFrame(tweet_with_id_list, columns=['TweetID','Tweet'])





# 5. Create bigram frequency table for tweet id based on bigrams
stopwords = nltk.corpus.stopwords.words('english')
stopWords = set(stopwords)
bigram_freq_dict = dict() 
bigram_tweetid_dict = dict()
bigram_tweetid_list = []
bigram_freq_list = []



print("Creating bigram frequency tables for tweetid's and tweets based on bigrams of words")
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
				bigram_tweetid_list.append(tweetID)
		else:
			bigram_freq_dict[forwardVar] = 1
			bigram_freq_dict[backwardVar] = 1
			bigram_tweetid_dict[forwardVar1] = 1
			bigram_tweetid_dict[backwardVar1] = 1
			bigram_tweetid_list.append(tweetID)



bfd = bigram_freq_dict
btd = bigram_tweetid_dict 
# Print results of bigram freq table
# for x in bft:
# 	print("Frequency: %d, Bigram: %s" % (bft[x],x))

# for x in btd:
# 	print("Frequency: %d, TweetID: %s" % (btd[x],x))

# Go through a tweet and look through the bigrams 
# Dictionary with highest frequency tweet for each bigram 
# To the summary, add the tweet for the highest frequency tweets present for each bigram



# for i in range(len(trump_tweet_list)-2):
# 	if(i==len(trump_tweet_list)-2):
# 		break

maxTweetIDs = [] 
summary_len = 0 
tweet_words = word_tokenize(test_text_trump)
numBigrams = len(tweet_words)*2


numBigrams= len(tweet_words)*2
print("Number of bigrams to compute: %d" % (numBigrams))
bigramCount = 0 
print("Computing bigram frequency for the dataset of tweets...")
max_freq = 0 



# Add tweet with highest frequency for those two words 
# Swap it out with a different one if you find another one later 

tweetIDFreqs = dict()
# Current tweet id freq
# Add up frequencies 
bigram_freq = 1
current_tweet_id = ""

for tweetidvar in bigram_tweetid_list:
	bigram_freq = 1 
	bigramCount += 2 
	current_freq = 0
	current_tweet_id = tweetidvar 
	for j in range(len(tweet_words)-1):
		wordA = tweet_words[j]
		wordA.lstrip()
		wordA.rstrip()
		wordB = tweet_words[j+1]
		wordB.lstrip()
		wordB.rstrip()
		bfstrA = wordA + " " + wordB 
		bfstrB = wordB + " " + wordA  
		if(bfstrA in bfd):
			bfstr1 = str(wordA + " " + wordB + " " + str(tweetidvar))
			bfstr2 = str(wordB + " " + wordA  + " " + str(tweetidvar))
			if(bfstr1 in btd):
				freqToAdd = btd[bfstr1]
				bigram_freq += freqToAdd 
			elif(bfstr2 in bfd):
				freqToAdd = bfd[bfstr2]
				bigram_freq += freqToAdd
	# Store them in dict 
	tweetIDFreqs[current_tweet_id] = bigram_freq

tweet_count = 0 
frequency_sum = 0 

def getTweetFromTweetID(tweet_id_var):
	tweetRow = tweet_with_id_df.loc[tweet_with_id_df['TweetID'] == tweet_id_var]
	tweetRowList = tweetRow['Tweet'].values.tolist()
	currentTweet = tweetRowList[0]
	return currentTweet 



for tweet_id, frequency in tweetIDFreqs.items():
 	frequency_sum += frequency 
 	tweet_count += 1

print("Extracted most relevant tweetids based on bigram frequency...")
print("Generating summary...")	
summary = ""

avg_frequency = frequency_sum / tweet_count 
threshold = 4

numTweetsAdded = 0 
for tweet_id, frequency in sorted(tweetIDFreqs.items(), key=lambda item: item[1]):
	if(frequency > (avg_frequency * threshold)):
		summary += getTweetFromTweetID(tweet_id)
		summary += " "
		numTweetsAdded += 1

print("Original Set of Tweets: ")
print(trump_tweet_text)
print("\n")
print("Summary: ")
print(summary)
print("\n")

print(numTweetsAdded)
# Bigram tweet ID freqs 
tweetIDFreqs = pd.DataFrame(list(tweetIDFreqs.items()), columns=['Tweet_ID','Frequency'])
























