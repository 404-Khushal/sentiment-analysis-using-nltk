# === IMPORT LIBRARIES ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag, ne_chunk
from tqdm.notebook import tqdm

# === DOWNLOAD NLTK RESOURCES ===
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# === LOAD DATA ===
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Reviews.csv')
df = df[['Id', 'Score', 'Text']].dropna().head(500)
print(f"Dataset loaded: {df.shape}")

# === EXPLORATORY DATA ANALYSIS ===
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Score', palette='viridis')
plt.title('Count of Reviews by Star Rating')
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.show()

# === SAMPLE REVIEW: POS + NER ===
sample_text = df['Text'].iloc[50]
print("Sample review:\n", sample_text)

tokens = word_tokenize(sample_text)
tags = pos_tag(tokens)
entities = ne_chunk(tags)

print("\nPart-of-Speech Tags:")
print(tags[:10])

print("\nNamed Entities:")
entities.pprint()

# === VADER SENTIMENT ANALYSIS ===
sia = SentimentIntensityAnalyzer()

vader_results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    score = sia.polarity_scores(text)
    compound = score['compound']
    
    sentiment = 'Neutral'
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    
    vader_results.append({
        'Id': row['Id'],
        'Text': text,
        'Score': row['Score'],
        'VADER_Compound': compound,
        'VADER_Sentiment': sentiment
    })

vader_df = pd.DataFrame(vader_results)
print("✅ VADER sentiment analysis complete.")
vader_df.head()

# === VISUALIZATION: VADER Compound Score by Star ===
plt.figure(figsize=(8, 5))
sns.barplot(data=vader_df, x='Score', y='VADER_Compound', palette='coolwarm')
plt.title('Average VADER Compound Score by Star Rating')
plt.xlabel('Star Rating')
plt.ylabel('Compound Sentiment Score')
plt.show()

# === EXTREME CASES: Sentiment Mismatch ===
print("\n1-Star Review with High Positive Sentiment:")
print(vader_df.query("Score == 1").sort_values("VADER_Compound", ascending=False)['Text'].values[0])

print("\n5-Star Review with High Negative Sentiment:")
print(vader_df.query("Score == 5").sort_values("VADER_Compound")['Text'].values[0])

# === CUSTOM USER INPUT ===
def run_custom_vader():
    print("\nCustom Sentiment Analyzer — type 'exit' to quit.")
    while True:
        text = input("\nEnter your review:\n")
        if text.lower() == 'exit':
            break
        score = sia.polarity_scores(text)
        compound = score['compound']
        sentiment = 'Neutral'
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        print(f"[VADER] Sentiment: {sentiment} | Compound Score: {compound:.3f}")
