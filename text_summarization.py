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

# Read datasets
trumpData = pd.read_csv("trumpImpeachment.csv")
natsData = pd.read_csv("washingtonNationals.csv")
redskinsData = pd.read_csv("washingtonRedskins.csv")
dcMetroData = pd.read_csv("dcMetro.csv")

# Drop duplicates and NA values
trumpData.drop_duplicates(subset=['Tweet'],inplace=True) 
trumpData.dropna(axis=0,inplace=True) 

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


# print(data1['Tweet'][:10])

stop_words = set(stopwords.words('english'))

def text_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

cleaned_text_trump = []
i = 0 
for t in trumpData['Tweet']:
	cleaned_text_trump.append(text_cleaner(t))
	i += 1 
	if i == 100:
		break


def listToString(s):
	str1 = ""

	for ele in s:
		str1 += ele 
		str1 += "\n"
	return str1



# print(trumpData['Tweet'][:10])

# Add summmary to data 
def create_frequency_table(text_string) -> dict:
	stopWords = set(stopwords.words("english"))
	words = word_tokenize(text_string)
	ps = PorterStemmer()

	freqTable = dict()
	for word in words:
		word = ps.stem(word)
		if word in stopWords:
			continue 
		if word in freqTable:
			freqTable[word] += 1
		else:
			freqTable[word] = 1 
	return freqTable 

def score_sentences(sentences,freqTable) -> dict:
	sentenceValue = dict()
	for sentence in sentences:
		word_count_in_sentence = (len(word_tokenize(sentence)))
		for wordValue in freqTable:
			if wordValue in sentence.lower():
				if sentence[:10] in sentenceValue:
					sentenceValue[sentence[:10]] += freqTable[wordValue]
				else:
					sentenceValue[sentence[:10]] = freqTable[wordValue]

		sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

	return sentenceValue 

def find_average_score(sentenceValue) -> int: 
	sumValues = 0
	for entry in sentenceValue:
		sumValues += sentenceValue[entry]

	# Average value of a sentence from original text 
	average = int(sumValues / len(sentenceValue))

	return average 

def generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1
            # print("Sentence %d \n" % (sentence_count) )

    return summary



def listToString(s):
	str1 = ""

	for ele in s:
		str1 += ele 
		str1 += "\n"
	return str1

cleaned_string_trump = listToString(cleaned_text_trump)


# print(cleaned_string_trump)

def allTweetsSummaryGenerator(cleanedTweets):
	# Create word frequency table 
	freq_table = create_frequency_table(cleanedTweets)
	# Tokenize sentences 
	sentences = sent_tokenize(cleanedTweets)
	# Score the sentences 
	sentence_scores = score_sentences(sentences,freq_table)
	# Find threshold 
	threshold = find_average_score(sentence_scores)
	# Generate summary 
	summary = generate_summary(sentences, sentence_scores, 1.5*threshold )
	return summary


# trump_summary = allTweetsSummaryGenerator(cleaned_string_trump)

# print(trump_summary)

# cleaned_string_trump = cleaned_string_trump.strip()
# cleaned_string_trump = " ".join(cleaned_string_trump.split())
s = cleaned_string_trump

 
s = re.sub(r"^\s+|\s+$", "", s)

# print("Summary Begin: %s\n" % (summarize(s)))
# print("Summary End:")

# print(cleaned_string_trump)


# trump_summary = allTweetsSummaryGenerator(s)

# print(trump_summary)

file = open('sampletext.txt',mode='r')

text = file.read()

file.close()
# textSum = allTweetsSummaryGenerator(text)
# print(textSum)


# Summary generation implemented without library 
def summaryGenerator(data):
	summaries = [] 
	tweets = [] 
	limit = 0 
	for tweet in data:
		# Clean tweet 
		tweet = tweet + "\n"
		tweet += "Donald Trump"
		tweet += "\n"
		# tweet += "Impeachment"
		# tweet += "\n"
		cleanedTweet = tweet
		words = set(nltk.corpus.words.words())
		cleanedTweet = " ".join(w for w in nltk.wordpunct_tokenize(cleanedTweet) \
         if w.lower() in words or not w.isalpha())
		# Create word frequency table 
		freq_table = create_frequency_table(cleanedTweet)
		# Tokenize sentences 
		sentences = sent_tokenize(cleanedTweet)
		# Score the sentences 
		sentence_scores = score_sentences(sentences,freq_table)
		# Find threshold 
		threshold = find_average_score(sentence_scores)
		# Generate summary 
		summary = generate_summary(sentences, sentence_scores, 1.5*threshold )
		# Add summary 
		word = 'Donald Trump'
		word_list = summary.split();
		summary = ' '.join([i for i in word_list if i not in word])
		summary = summary.strip()
		if(len(summary) > 3):
			summaries.append(summary)
			tweets.append(tweet)
			limit += 1 
		if limit == 100:
			break

	# summaries = pd.DataFrame(list(zip(tweets, summaries)), columns =['Tweet', 'Summary']) 

	return summaries

# Print tweet summaries
def summary_cleaner(text):
	newString = re.sub('"','', text)
	newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
	newString = re.sub(r"'s\b","",newString)
	newString = re.sub("[^a-zA-Z]"," ",newString)
	newString = newString.lower()
	tokens = newString.split()
	newString = ''
	for i in tokens:
		if len(i)>1:
			newString = newString + i + ' '
	return newString 





# Summarization using summarize function in nltk
def summaryGenerator2(data):
	i = 0 
	summaries = []

	for tweet in data:
		# tweet = text_cleaner(tweet)
		tweet = tweet + "\n"
		# tweet += "Donald Trump"
		tweet += "\n"
		# tweet += "Impeachment"
		tweet += "\n"
		summary = summarize(tweet)
		if(len(summary) > 5 and summary != ''):
			i+= 1
			summaries.append(summary)
		if i == 100:
			break
	return summaries 

# Get summaries 
# trumpSummaries = summaryGenerator(trumpData['Tweet'])



summaries = [] 
tweets = [] 
i = 0 
for tweet in trumpData['Tweet']:
	# Clean tweet 
	tweet = tweet + "\n"
	tweet += "Donald Trump"
	tweet += "\n"
	# tweet += "Impeachment"
	# tweet += "\n"
	cleanedTweet = tweet
	words = set(nltk.corpus.words.words())
	cleanedTweet = " ".join(w for w in nltk.wordpunct_tokenize(cleanedTweet) \
		if w.lower() in words or not w.isalpha())
	# Create word frequency table 
	freq_table = create_frequency_table(cleanedTweet)
	# Tokenize sentences 
	sentences = sent_tokenize(cleanedTweet)
	# Score the sentences 
	sentence_scores = score_sentences(sentences,freq_table)
	# Find threshold 
	threshold = find_average_score(sentence_scores)
	# Generate summary 
	summary = generate_summary(sentences, sentence_scores, 1.5*threshold )
	# Add summary 
	word = 'Donald Trump'
	word_list = summary.split();
	summary = ' '.join([i for i in word_list if i not in word])
	tweet = ' '.join([i for i in word_list if i not in word])
	summary = summary.strip()
	if(len(summary) > 3):
		summaries.append(summary)
		tweets.append(tweet)
		i = i+1
	if i == 10:
		break



for i in range(len(summaries)):
	print("Tweet: ")
	print(tweets[i])
	print("Summary: ")
	print(summary_cleaner(summaries[i]))
	print("\n")

listDF = list(zip(tweets, summaries))  

trumpDF = pd.DataFrame(list_of_tuples, columns = ['Name', 'Age']
trumpDF[]
# tweetNum = 0
# for i in range(len(summaries)):
# 	tweetNum += 1 
# 	print("Tweet %d: %s \n" % (tweetNum,tweets[i]))
#  	print("Summary: %s: \n" % (summary_cleaner(summaries[i])))




# tweetNum = 0
# for tweet, summary in trumpSummaries:
#  	tweetNum += 1 
#  	print("Tweet: %s \n " % (tweet))
#  	print("Tweet Summary: %d " % (tweetNum))
#  	print(summary_cleaner(summary))
#  	# print(summary_cleaner(summary))
#  	print("\n")














