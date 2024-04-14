import finnhub
from datetime import date, timedelta
from transformers import pipeline
import pandas as pd

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
yesterday = today - timedelta(days=1)

formatted_today = today.strftime('%Y-%m-%d')
formatted_yesterday = yesterday.strftime('%Y-%m-%d')


def classify_sentiment(text):

    # Load the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Custom labels for financial sentiment analysis
    candidate_labels = ["Slight Negative", "Very Negative", "Slight Positive", "Very Positive", "Neutral"]
    
    # Perform zero-shot classification
    result = classifier(text, candidate_labels)

    if result['labels'][0] == 'Slight Negative': 
        return 'Slight Negative 😟'
    elif result['labels'][0] == 'Very Negative':
        return 'Very Negative 😡'
    elif result['labels'][0] == 'Slight Positive':
        return 'Slight Positive 🙂'
    elif result['labels'][0] == 'Very Positive':
        return 'Very Positive 😁'
    elif result['labels'][0] == 'Neutral':
        return 'Neutral 😐'



def get_stock_news(ticker_symbol):
    news = finnhub_client.company_news(ticker_symbol, _from=yesterday, to=formatted_today)
    df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
    top_5_news = df.head(5)
    top_5_news['stand'] = top_5_news['summary'].apply(classify_sentiment)
    top_5_news = top_5_news.reset_index(drop=True)
    return top_5_news
