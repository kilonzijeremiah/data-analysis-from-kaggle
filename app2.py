from newspaper import Article, Config
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

config = Config()
config.browser_user_agent = "Mozilla/5.0"

url = "https://www.nationmedia.com/news/business-daily-fetes-top-40-40-men-nairobi-gala/"

article = Article(url, config=config)
article.download()
article.parse()

st.title(article.title)
st.write(article.text[:500])

# FIX: define soup properly
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.text, "html.parser")

text = " ".join([p.get_text() for p in soup.find_all("p")])

analysis = TextBlob(text)

st.write("Polarity:", analysis.sentiment.polarity)
st.write("Subjectivity:", analysis.sentiment.subjectivity)

analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)

st.write("Sentiment Scores:", scores)

if scores['compound'] >= 0.05:
    st.write("Sentiment: Positive 😊")
elif scores['compound'] <= -0.05:
    st.write("Sentiment: Negative 😞")
else:
    st.write("Sentiment: Neutral 😐")

import pandas as pd
import matplotlib.pyplot as plt

# FIX: create dataframe for plotting
df = pd.DataFrame([scores])

fig, ax = plt.subplots()
df[['pos', 'neg', 'neu']].iloc[0].plot(kind='bar', ax=ax)

st.pyplot(fig)