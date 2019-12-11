# # cleaned_string_trump = cleaned_string_trump.strip()
# # cleaned_string_trump = " ".join(cleaned_string_trump.split())
# s = cleaned_string_trump

 
# s = re.sub(r"^\s+|\s+$", "", s)

# # print("Summary Begin: %s\n" % (summarize(s)))
# # print("Summary End:")

# # print(cleaned_string_trump)


# # trump_summary = allTweetsSummaryGenerator(s)

# # print(trump_summary)

# file = open('sampletext.txt',mode='r')

# text = file.read()

# file.close()
# # textSum = allTweetsSummaryGenerator(text)
# # print(textSum)


# # Summary generation implemented without library 
# def summaryGenerator(data):
# 	summaries = [] 
# 	tweets = [] 
# 	limit = 0 
# 	for tweet in data:
# 		# Clean tweet 
# 		tweet = tweet + "\n"
# 		tweet += "Donald Trump"
# 		tweet += "\n"
# 		# tweet += "Impeachment"
# 		# tweet += "\n"
# 		cleanedTweet = tweet
# 		words = set(nltk.corpus.words.words())
# 		cleanedTweet = " ".join(w for w in nltk.wordpunct_tokenize(cleanedTweet) \
#          if w.lower() in words or not w.isalpha())
# 		# Create word frequency table 
# 		freq_table = create_frequency_table(cleanedTweet)
# 		# Tokenize sentences 
# 		sentences = sent_tokenize(cleanedTweet)
# 		# Score the sentences 
# 		sentence_scores = score_sentences(sentences,freq_table)
# 		# Find threshold 
# 		threshold = find_average_score(sentence_scores)
# 		# Generate summary 
# 		summary = generate_summary(sentences, sentence_scores, 1.5*threshold )
# 		# Add summary 
# 		word = 'Donald Trump'
# 		word_list = summary.split();
# 		summary = ' '.join([i for i in word_list if i not in word])
# 		summary = summary.strip()
# 		if(len(summary) > 3):
# 			summaries.append(summary)
# 			tweets.append(tweet)
# 			limit += 1 
# 		if limit == 100:
# 			break

# 	# summaries = pd.DataFrame(list(zip(tweets, summaries)), columns =['Tweet', 'Summary']) 

# 	return summaries

# # Print tweet summaries
# def summary_cleaner(text):
# 	newString = re.sub('"','', text)
# 	newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
# 	newString = re.sub(r"'s\b","",newString)
# 	newString = re.sub("[^a-zA-Z]"," ",newString)
# 	newString = newString.lower()
# 	tokens = newString.split()
# 	newString = ''
# 	for i in tokens:
# 		if len(i)>1:
# 			newString = newString + i + ' '
# 	return newString 





# # Summarization using summarize function in nltk
# def summaryGenerator2(data):
# 	i = 0 
# 	summaries = []

# 	for tweet in data:
# 		# tweet = text_cleaner(tweet)
# 		tweet = tweet + "\n"
# 		# tweet += "Donald Trump"
# 		tweet += "\n"
# 		# tweet += "Impeachment"
# 		tweet += "\n"
# 		summary = summarize(tweet)
# 		if(len(summary) > 5 and summary != ''):
# 			i+= 1
# 			summaries.append(summary)
# 		if i == 100:
# 			break
# 	return summaries 

# # Get summaries 
# # trumpSummaries = summaryGenerator(trumpData['Tweet'])



# summaries = [] 
# tweets = [] 
# i = 0 
# for tweet in trumpData['Tweet']:
# 	# Clean tweet 
# 	tweet = tweet + "\n"
# 	tweet += "Donald Trump"
# 	tweet += "\n"
# 	# tweet += "Impeachment"
# 	# tweet += "\n"
# 	cleanedTweet = tweet
# 	words = set(nltk.corpus.words.words())
# 	cleanedTweet = " ".join(w for w in nltk.wordpunct_tokenize(cleanedTweet) \
# 		if w.lower() in words or not w.isalpha())
# 	# Create word frequency table 
# 	freq_table = create_frequency_table(cleanedTweet)
# 	# Tokenize sentences 
# 	sentences = sent_tokenize(cleanedTweet)
# 	# Score the sentences 
# 	sentence_scores = score_sentences(sentences,freq_table)
# 	# Find threshold 
# 	threshold = find_average_score(sentence_scores)
# 	# Generate summary 
# 	summary = generate_summary(sentences, sentence_scores, 1.5*threshold )
# 	# Add summary 
# 	word = 'Donald Trump'
# 	word_list = summary.split();
# 	summary = ' '.join([i for i in word_list if i not in word])
# 	tweet = ' '.join([i for i in word_list if i not in word])
# 	summary = summary.strip()
# 	if(len(summary) > 3):
# 		summaries.append(summary)
# 		tweets.append(tweet)
# 		i = i+1
# 	if i == 10:
# 		break



# for i in range(len(summaries)):
# 	print("Tweet: ")
# 	print(tweets[i])
# 	print("Summary: ")
# 	print(summary_cleaner(summaries[i]))
# 	print("\n")

# listDF = list(zip(tweets, summaries))  

# trumpDF = pd.DataFrame(list_of_tuples, columns = ['Name', 'Age'])

# tweetNum = 0
# for i in range(len(summaries)):
# 	tweetNum += 1 
# 	print("Tweet %d: %s \n" % (tweetNum,tweets[i]))
# 	print("Summary: %s: \n" % (summary_cleaner(summaries[i])))




# tweetNum = 0
# for tweet, summary in trumpSummaries:
#  	tweetNum += 1 
#  	print("Tweet: %s \n " % (tweet))
#  	print("Tweet Summary: %d " % (tweetNum))
#  	print(summary_cleaner(summary))
#  	# print(summary_cleaner(summary))
#  	print("\n")














