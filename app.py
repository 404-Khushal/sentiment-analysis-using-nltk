# app.py

import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import word_tokenize
from wordcloud import WordCloud
import string

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# === Load Dataset ===
@st.cache_data
def load_data():
    df = pd.read_csv('./reviews.csv')
    df = df[['Id', 'Score', 'Text']].dropna().head(1000)
    return df

# === Clean Text ===
def clean_text(text):
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    cleaned = [w for w in words if w not in stop_words]
    return ' '.join(cleaned)

# === VADER Analysis ===
def analyze_sentiment(text, sia):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        sentiment = 'Positive'
    elif score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, score

# === Generate Word Cloud ===
def generate_wordcloud(texts, title):
    text_blob = ' '.join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.subheader(title)
    st.pyplot(fig)

# === Main App ===
def main():
    st.title("Sentiment Analysis with NLTK (VADER)")
    st.markdown("Analyze Amazon reviews and your own custom input using NLTK’s rule-based sentiment engine (VADER).")

    df = load_data()
    sia = SentimentIntensityAnalyzer()

    tab1, tab2, tab3 = st.tabs(["📊 Dataset Analysis", "📝 Custom Input", "☁️ Word Clouds"])

    # === Tab 1: Dataset Analysis ===
    with tab1:
        st.subheader("Amazon Reviews Sample (500 rows)")
        st.dataframe(df[['Score', 'Text']].sample(5))

        st.subheader("Sentiment Breakdown")
        results = []
        for _, row in df.iterrows():
            cleaned = clean_text(row['Text'])
            sentiment, comp = analyze_sentiment(cleaned, sia)
            results.append({'Id': row['Id'], 'Score': row['Score'], 'Cleaned_Text': cleaned, 'Sentiment': sentiment, 'Compound': comp})
        results_df = pd.DataFrame(results)
        st.bar_chart(results_df['Sentiment'].value_counts())

        st.subheader("VADER Compound Score vs Star Rating")
        sns.barplot(data=results_df, x='Score', y='Compound')
        st.pyplot(plt.gcf())
        plt.clf()

    # === Tab 2: Custom Input ===
    with tab2:
        st.subheader("Type a Review to Analyze")
        user_text = st.text_area("Enter your review here:", height=150)
        if user_text:
            cleaned = clean_text(user_text)
            sentiment, score = analyze_sentiment(cleaned, sia)
            st.markdown(f"**Cleaned Text:** {cleaned}")
            st.success(f"Sentiment: **{sentiment}** | Compound Score: **{score:.3f}**")

    # === Tab 3: Word Cloud ===
    with tab3:
        st.subheader("Word Clouds by Star Rating")
        col1, col2 = st.columns(2)

        with col1:
            generate_wordcloud(df[df['Score'] == 1]['Text'].apply(clean_text), "1-Star Reviews")

        with col2:
            generate_wordcloud(df[df['Score'] == 5]['Text'].apply(clean_text), "5-Star Reviews")

if __name__ == "__main__":
    main()
