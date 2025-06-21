# sentiment_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os

# Set display options for better viewing
pd.set_option('display.max_columns', None)

# Step 1: Load the dataset
DATA_PATH = 'Twitter_Data.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset '{DATA_PATH}' not found!")

df = pd.read_csv(DATA_PATH)

# Step 2: Check and clean data
print("Original dataset shape:", df.shape)
df.dropna(subset=['clean_text'], inplace=True)
print("After dropping nulls in 'clean_text':", df.shape)

# Step 3: Sentiment classification function
def get_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    except:
        return 'Neutral'  # fallback in case of errors

# Step 4: Apply sentiment analysis
print("Analyzing sentiment...")
df['Sentiment'] = df['clean_text'].apply(get_sentiment)

# Step 5: Save the results to CSV
df.to_csv('Twitter_Sentiment_Results.csv', index=False)
print("Results saved to 'Twitter_Sentiment_Results.csv'")

# Step 6: Plot sentiment distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Sentiment', palette='pastel')
plt.title('Sentiment Distribution in Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()
print("Bar chart saved as 'sentiment_distribution.png'")
