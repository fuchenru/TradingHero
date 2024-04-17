import finnhub
from datetime import date, timedelta
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

import pandas as pd

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

today = date.today()
yesterday = today - timedelta(weeks=3)

formatted_today = today.strftime('%Y-%m-%d')
formatted_yesterday = yesterday.strftime('%Y-%m-%d')


def classify_sentiment(text):

    # Load the zero-shot classification pipeline
    classifier = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    
    # Perform zero-shot classification
    result = classifier(text)

    if result[0]['label'] == 'Negative': 
        return 'Negative 😡'
    elif result[0]['label'] == 'Positive':
        return 'Positive 😁'
    elif result[0]['label'] == 'Neutral':
        return 'Neutral 😐'



def get_stock_news(ticker_symbol):
    news = finnhub_client.company_news(ticker_symbol, _from=yesterday, to=formatted_today)
    df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
    top_news = df[~df['summary'].str.contains('Looking for stock market analysis and research with proves results?')]
    top_news = top_news.head(100)
    top_news['stand'] = top_news['summary'].apply(classify_sentiment)
    top_news = top_news.reset_index(drop=True)
    return top_news
