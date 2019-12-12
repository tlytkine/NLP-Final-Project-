# Import libraries 
from attention import AttentionLayer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize, sent_tokenize 
import numpy as np 
import pandas as pd 
import re 
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from nltk.corpus import stopwords 
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings 
pd.set_option("display.max_colwidth",200)
warnings.filterwarnings("ignore")
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords 
from urllib.request import urlopen
import nltk
import requests 
import csv
import matplotlib.pyplot as plt
from collections import Counter
from heapq import nlargest 
import string 
from string import digits

# Read datasets 
trumpData = pd.read_csv('trumpImpeachment.csv',engine='python',header='infer',names=['Tweet','id','date','source','likes','retweets','sentiment'])
# natsData = pd.read_csv('washingtonNationals.csv',engine='python',header='infer',names=['Tweet','id','date','source','likes','retweets','sentiment'])
# redskinsData = pd.read_csv('washingtonRedskins.csv',engine='python',header='infer',names=['Tweet','id','date','source','likes','retweets','sentiment'])
# dcMetroData = pd.read_csv('dcMetro.csv',engine='python',header='infer',names=['Tweet','id','date','source','likes','retweets','sentiment'])


# Drop duplicates and NA values

# Trump Data 
trumpData.drop_duplicates(subset=['Tweet'],inplace=True) 
trumpData.dropna(axis=0,inplace=True) 
trumpData["retweets"] = pd.to_numeric(trumpData["retweets"],errors='coerce')
trumpData = trumpData.nlargest(100, columns=['retweets'])
# Nats Data 
# natsData.drop_duplicates(subset=['Tweet'],inplace=True) 
# natsData.dropna(axis=0,inplace=True) 
# natsData["retweets"] = pd.to_numeric(natsData["retweets"],errors='coerce')
# natsData = natsData.nlargest(100, columns=['retweets'])
# # Redskins data 
# redskinsData.drop_duplicates(subset=['Tweet'],inplace=True) 
# redskinsData.dropna(axis=0,inplace=True) 
# redskinsData["retweets"] = pd.to_numeric(redskinsData["retweets"],errors='coerce')
# redskinsData = redskinsData.nlargest(100, columns=['retweets'])
# # Dc metro data
# dcMetroData.drop_duplicates(subset=['Tweet'],inplace=True) 
# dcMetroData.dropna(axis=0,inplace=True) 
# dcMetroData["retweets"] = pd.to_numeric(dcMetroData["retweets"],errors='coerce')
# dcMetroData = dcMetroData.nlargest(100, columns=['retweets'])



# Preprocessing 

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


text_trump = []

for t in trumpData['Tweet']:
	t = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', t)
	t = re.sub('\W+',' ', t)
	remove_digits = str.maketrans('', '', digits)
	t = t.translate(remove_digits)
	# shortword = re.compile(r'\W*\b\w{1,3}\b')
	# shortword.sub('',t)
	# words = set(nltk.corpus.words.words())
	# t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
	text_trump.append(t)

# text_nats = []

# for t in natsData['Tweet']:
# 	t = re.sub('\W+',' ', t)
# 	shortword = re.compile(r'\W*\b\w{1,3}\b')
# 	shortword.sub('',t)
# 	words = set(nltk.corpus.words.words())
# 	t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
# 	text_nats.append(t)

# text_redskins = []

# for t in redskinsData['Tweet']:
# 	t = re.sub('\W+',' ', t)
# 	shortword = re.compile(r'\W*\b\w{1,3}\b')
# 	shortword.sub('',t)
# 	words = set(nltk.corpus.words.words())
# 	t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
# 	text_redskins.append(t)

# text_dcMetro = []

# for t in dcMetroData['Tweet']:
# 	t = re.sub('\W+',' ', t)
# 	shortword = re.compile(r'\W*\b\w{1,3}\b')
# 	shortword.sub('',t)
# 	words = set(nltk.corpus.words.words())
# 	t = " ".join(w for w in nltk.wordpunct_tokenize(t) if w.lower() in words or not w.isalpha())
# 	text_dcMetro.append(t)

# Create word frequency table from text 
# Does not include stopwords 
def create_frequency_table(text_string) -> dict:
	stopwords = nltk.corpus.stopwords.words('english')
	stopWords = set(stopwords)

	words = word_tokenize(text_string)
	freqTable = dict()
	for word in words:
		word = word.lower()
		if word in stopWords:
			continue 
		if(len(word) < 3):
			continue
		if word in freqTable:
			freqTable[word] += 1
		else:
			freqTable[word] = 1 
	return freqTable 


cleanedTrumpTweets = " ".join(text_trump)


trumpFreqTable = create_frequency_table(cleanedTrumpTweets)

ten_largest_trump = nlargest(10,trumpFreqTable, key=trumpFreqTable.get)

trumpFreqTableNew = dict()
for word in ten_largest_trump:
 	if word in trumpFreqTable:
 		trumpFreqTableNew[word] = trumpFreqTable[word]

# plt.bar(range(len(trumpFreqTableNew)), list(trumpFreqTableNew.values()), align='center')
# plt.xticks(range(len(trumpFreqTableNew)), list(trumpFreqTableNew.keys()), rotation='vertical')
# plt.tight_layout()
# plt.show()

# cleanedNatsTweets = " ".join(text_nats)


# natsFreqTable = create_frequency_table(cleanedNatsTweets)

# ten_largest_nats = nlargest(10,natsFreqTable, key=natsFreqTable.get)

# natsFreqTableNew = dict()
# for word in ten_largest_nats:
#  	if word in natsFreqTable:
#  		natsFreqTableNew[word] = natsFreqTable[word]


# plt.bar(range(len(natsFreqTableNew)), list(natsFreqTableNew.values()), align='center')
# plt.xticks(range(len(natsFreqTableNew)), list(natsFreqTableNew.keys()), rotation='vertical')
# plt.tight_layout()
# plt.show()

# cleanedRedskinsTweets = " ".join(text_redskins)


# redskinsFreqTable = create_frequency_table(cleanedRedskinsTweets)

# ten_largest_redskins = nlargest(10,redskinsFreqTable, key=redskinsFreqTable.get)

# redskinsFreqTableNew = dict()
# for word in ten_largest_redskins:
#  	if word in redskinsFreqTable:
#  		redskinsFreqTableNew[word] = redskinsFreqTable[word]

# plt.bar(range(len(redskinsFreqTableNew)), list(redskinsFreqTableNew.values()), align='center')
# plt.xticks(range(len(redskinsFreqTableNew)), list(redskinsFreqTableNew.keys()), rotation='vertical')
# plt.tight_layout()
# plt.show()

# cleanedDCmetroTweets = " ".join(text_dcMetro)


# dcMetroFreqTable = create_frequency_table(cleanedDCmetroTweets)

# ten_largest_dcMetro = nlargest(10,dcMetroFreqTable, key=dcMetroFreqTable.get)

# dcMetroFreqTableNew = dict()
# for word in ten_largest_dcMetro:
#  	if word in dcMetroFreqTable:
#  		dcMetroFreqTableNew[word] = dcMetroFreqTable[word]


# plt.bar(range(len(dcMetroFreqTableNew)), list(dcMetroFreqTableNew.values()), align='center')
# plt.xticks(range(len(dcMetroFreqTableNew)), list(dcMetroFreqTableNew.keys()), rotation='vertical')
# plt.tight_layout()
# plt.show()





def find_average_score(sentenceValue) -> int: 
	sumValues = 0
	for entry in sentenceValue:
		sumValues += sentenceValue[entry]

	# Average value of a sentence from original text 
	average = int(sumValues / len(sentenceValue))

	return average 

tenSentText = ""

i = 0
# Get 10 sentences 
for tweet in text_trump:
	if(i==10):
		break
	tweet = ' '.join(tweet.split())
	toAdd = ' '.join( [w for w in tweet.split() if len(w)>1] )
	toAdd.lstrip()
	toAdd.rstrip()
	tenSentText += " " + toAdd.lower()
	i = i+1


tenSent = []

i = 0
# Get 10 sentences 
for tweet in text_trump:
	if(i==10):
		break
	tweet = ' '.join(tweet.split())
	toAdd = ' '.join( [w for w in tweet.split() if len(w)>1] )
	toAdd.lstrip()
	toAdd.rstrip()
	tenSent.append(toAdd.lower())
	i = i+1


# Create word frequency table 
freq_table = create_frequency_table(tenSentText)
# Tokenize sentences 
sentences = sent_tokenize(tenSentText)
# Score Sentences 
print("Scoring sentences: ")
sentenceValue = dict()
for sentence in tenSent:
	print("Sentence: ")
	print(sentence)
	word_count_in_sentence = (len(word_tokenize(sentence)))
	for wordValue in freq_table:
		if wordValue in sentence.lower():
				if sentence[:10] in sentenceValue:
					sentenceValue[sentence[:10]] += freq_table[wordValue]
				else:
					sentenceValue[sentence[:10]] = freq_table[wordValue]
		print("sentence[:10]: ")
		print(sentence[:10])
		if(sentence[:10] in sentenceValue):
			sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence
sentence_scores = sentenceValue 
threshold = find_average_score(sentence_scores)

	
sentence_count = 0
summary = ''
for sentence in tenSent:
	if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
		summary += " " + sentence
		sentence_count += 1


# Print ten sentences 
print("Original: ")
print(tenSent)
print("Summary: ")
print(summary)

# 1. Get tweets in list 
# Generate bigrams 
# Put bigrams into dictionary with frequency for all tweets 
# Pick tweets who have the most bigrams 




def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]








# for tweet in text_trump:
# 	print("Summary: ")
# 	print(TweetSummaryGenerator(tweet))
	# trumpSummaries.append(TweetSummaryGenerator(tweet))

# print(trumpSummaries[0])



