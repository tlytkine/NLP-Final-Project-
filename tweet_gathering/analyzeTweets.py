import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from plotly.offline import plot, iplot

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

#get data
nats = pd.read_csv('washingtonNationals.csv')

#make wordcloud
# text = " ".join(tweet for tweet in nats.Tweet)
# stopwords = set(STOPWORDS)
# stopwords.update(['washington', 'nationals', 'national', 'https', 'co'])
#
# wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=100, background_color="white").generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
#
# wordcloud.to_file('img/nats.png')

#do some other analysis and visualization
print('5 most positive tweets:')
print(nats.nlargest(5, 'sentiment'))

print('5 most negative tweets:')
print(nats.nsmallest(5, 'sentiment'))

print('Average sentiment:')
print(np.mean(nats['sentiment']))

plt.hist(nats['sentiment'],bins=50)
plt.show()
