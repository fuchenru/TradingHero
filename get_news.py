import finnhub
from datetime import date
from textblob import TextBlob

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
formatted_today = today.strftime('%Y-%m-%d')

def classify_sentiment(text):
    blob = TextBlob(text)
    
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def get_stock_news(ticker_symbol):
    news = finnhub_client.company_news(ticker_symbol, _from=formatted_today, to=formatted_today)
    df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
    top_5_news = df.head(5)
    top_5_news['stand'] = top_5_news['summary'].apply(classify_sentiment)
    return top_5_news
