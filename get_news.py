import finnhub
from datetime import date
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
formatted_today = today.strftime('%Y-%m-%d')

def classify_sentiment(text):
    result = nlp(text)
    
    label = result[0]['label']
    
    if label == 'negative':
        return 'Negative 😡'
    elif label == 'neutral':
        return 'Neutral 😐'
    elif label == 'positive':
        return 'Positive 🙂'
    else:
        return 'Error: Unknown label'

def prediction_score(text):
    result = nlp(text)
    score = result[0]['score'] * 100
    return(f"{score:.2f}%")


# def get_stock_news(ticker_symbol):
#     news = finnhub_client.company_news(ticker_symbol, _from="2024-04-01", to=formatted_today)
#     df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
#     top_5_news = df.head(5)
#     top_5_news['Sentiment Analysis'] = top_5_news['summary'].apply(classify_sentiment) 
#     top_5_news['Score'] = top_5_news['summary'].apply(prediction_score)
#     top_5_news = top_5_news.reset_index(drop=True)
#     return top_5_news

def get_stock_news(ticker_symbol):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from="2024-04-01", to=formatted_today)
        df = pd.DataFrame(news)
        df = df[['headline', 'summary']].head(5)
        df.rename(columns={'headline': 'Headline', 'summary': 'Summary'}, inplace=True)
        df['Sentiment Analysis'] = df['Summary'].apply(classify_sentiment)
        df['Score'] = df['Summary'].apply(prediction_score)
        return df
    except Exception as e:
        return f"An error occurred: {e}"