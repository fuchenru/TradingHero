import finnhub
from datetime import date
from textblob import TextBlob
import pandas as pd

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
formatted_today = today.strftime('%Y-%m-%d')

def classify_sentiment(text):
    blob = TextBlob(text)
    
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return 'Positive 🙂'
    elif polarity < 0:
        return 'Negative ☹️'
    else:
        return 'Neutral 😐'

def get_stock_news(ticker_symbol):
    news = finnhub_client.company_news(ticker_symbol, _from="2024-04-01", to="2024-04-01")
    df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
    top_5_news = df.head(5)
    top_5_news['stand'] = top_5_news['summary'].apply(classify_sentiment)
    top_5_news = top_5_news.reset_index(drop=True)
    return top_5_news
