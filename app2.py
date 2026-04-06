import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📰 News Sentiment Analysis")

url = "https://www.nationmedia.com/news/business-daily-fetes-top-40-40-men-nairobi-gala/"

# ----------------------------
# SCRAPING
# ----------------------------
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")

text = " ".join([p.get_text() for p in soup.find_all("p")])

# Show article preview
st.subheader("📄 Article Preview")
st.write(text[:500])

# ----------------------------
# SENTIMENT ANALYSIS (VADER)
# ----------------------------
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)

# Display scores
st.subheader("📊 Sentiment Scores")
st.write(scores)

# Classification
if scores['compound'] >= 0.05:
    st.success("Sentiment: Positive 😊")
elif scores['compound'] <= -0.05:
    st.error("Sentiment: Negative 😞")
else:
    st.info("Sentiment: Neutral 😐")

# ----------------------------
# VISUALIZATION
# ----------------------------
st.subheader("📈 Sentiment Visualization")

df = pd.DataFrame([scores])

fig, ax = plt.subplots()
df[['pos', 'neu', 'neg']].iloc[0].plot(kind='bar', ax=ax)

ax.set_ylabel("Score")
ax.set_title("Sentiment Breakdown")

st.pyplot(fig)
