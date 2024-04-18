import finnhub
from datetime import date, timedelta
from transformers import pipeline

import pandas as pd
import re

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
prev_date = today - timedelta(weeks=3)

formatted_today = today.strftime('%Y-%m-%d')
formatted_prevdate = prev_date.strftime('%Y-%m-%d')


def classify_sentiment(text):

    # Load the zero-shot classification pipeline
    classifier = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    
    # Perform zero-shot classification
    result = classifier(text)

    if result[0]['label'] == 'negative': 
        return 'Negative 😡'
    elif result[0]['label'] == 'positive':
        return 'Positive 😁'
    elif result[0]['label'] == 'neutral':
        return 'Neutral 😐'



# Function to count AI-related words in a text
def count_related_words(text, ticker_symbol):
    # Find all occurrences of the AI-related keywords in the text
    matches = re.findall(ticker_symbol, text, flags=re.IGNORECASE)
    # Return the number of matches
    return len(matches)


def get_stock_news(ticker_symbol):
    news = finnhub_client.company_news(ticker_symbol, _from=formatted_prevdate, to=formatted_today)
    df = pd.DataFrame.from_records(news, columns=['headline', 'summary']).head(1000)

    # Drop those have no summary content
    df = df[~df['summary'].str.contains('Looking for stock market analysis and research with proves results?')]
    
    # Select words less than 500 and reset index by counting related word
    df['word_counts'] = df['summary'].str.split().str.len()
    df['related_word_count'] = df['summary'].apply(lambda text: count_related_words(text, ticker_symbol))

    df = df[(df['word_counts'] <= 500)]
    top_news = df.sort_values(by='related_word_count', ascending=False).head(100)

    top_news['stand'] = top_news['summary'].apply(classify_sentiment)
    top_news = top_news.reset_index(drop=True)
    return top_news[['headline', 'summary', 'stand']]
