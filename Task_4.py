import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load your dataset
df = pd.read_csv('twitter_training.csv', parse_dates=['timestamp'])

# Preprocessing (sample)
df['clean_text'] = df['text'].str.lower().str.replace(r'http\S+', '', regex=True)

# Sentiment scoring
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['clean_text'].apply(lambda t: sia.polarity_scores(t)['compound'])

# Classify sentiment
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda s: 'positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral'
)

# Aggregate by day
daily = df.set_index('timestamp').resample('D').agg({
    'sentiment_score': 'mean',
    'text': 'count'
}).rename(columns={'text': 'post_count'})

# Visualize trend
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(daily.index, daily['sentiment_score'], color='blue', label='Avg Sentiment')
ax1.set_ylabel('Avg Sentiment Score', color='blue')

ax2 = ax1.twinx()
ax2.bar(daily.index, daily['post_count'], alpha=0.3, color='gray', label='Post Count')
ax2.set_ylabel('Number of Posts', color='gray')

fig.legend(loc='upper right')
plt.title('Sentiment & Volume Over Time')
plt.show()
