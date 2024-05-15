# import finnhub
# from datetime import date, timedelta
# import pandas as pd
# import numpy as np
# import warnings
# import re
# warnings.filterwarnings('ignore')

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

# finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

# today = date.today()
# prev_date = today - timedelta(weeks=3)

# formatted_today = today.strftime('%Y-%m-%d')
# formatted_prevdate = prev_date.strftime('%Y-%m-%d')

# def classify_sentiment(text):
#     result = nlp(text)
    
#     label = result[0]['label']
    
#     if label == 'negative':
#         return 'Negative 😡'
#     elif label == 'neutral':
#         return 'Neutral 😐'
#     elif label == 'positive':
#         return 'Positive 🙂'
#     else:
#         return 'Error: Unknown label'

# def prediction_score(text):
#     result = nlp(text)
#     score = result[0]['score'] * 100
#     return(f"{score:.2f}%")


# def count_related_words(text, ticker_symbol):
#     # Find all occurrences of the AI-related keywords in the text
#     matches = re.findall(ticker_symbol, text, flags=re.IGNORECASE)
#     # Return the number of matches
#     return len(matches)

# def get_stock_news(ticker_symbol):
#     try:
#         news = finnhub_client.company_news(ticker_symbol, _from=formatted_prevdate, to=formatted_today)
#         df = pd.DataFrame(news)
#         df = df[['headline', 'summary']]
#         df = df[~df['summary'].str.contains('Looking for stock market analysis and research with proves results?')]
    
#         # Select words less than 500 and reset index by counting related word
#         df['word_counts'] = df['summary'].str.split().str.len()
#         df['related_word_count'] = df['summary'].apply(lambda text: count_related_words(text, ticker_symbol))

#         df = df[(df['word_counts'] <= 500)]

#         df.rename(columns={'headline': 'Headline', 'summary': 'Summary'}, inplace=True)
#         df['Sentiment Analysis'] = df['Summary'].apply(classify_sentiment)
#         df['Score'] = df['Summary'].apply(prediction_score)
#         df = df.drop(columns=['word_counts', 'related_word_count'])
#         return df.head(5)
#     except Exception as e:
#         return f"An error occurred: {e}"
import finnhub
from datetime import date
from textblob import TextBlob
import pandas as pd
import requests
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-flash-preview-0514")
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
formatted_today = today.strftime('%Y-%m-%d')

def classify_sentiment(text):
    blob = TextBlob(text)
    
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return 'Positive 🙂'
    elif polarity < 0:
        return 'Negative 😡'
    else:
        return 'Neutral 😐'

def get_stock_news(ticker_symbol, start_date, end_date):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
        if df.empty:
            return "No news available for the given date range."
        top_5_news = df.head(5)
        top_5_news['Sentiment Analysis'] = top_5_news['summary'].apply(classify_sentiment)
        top_5_news = top_5_news.reset_index(drop=True)
        return top_5_news
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    
def get_80_stock_news(ticker_symbol, start_date, end_date):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
        if df.empty:
            return "No news available for the given date range."
        top_80_news = df.head(80)
        return top_80_news['headline']
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    
def extract_text_from_response(response):
    # Check if there are candidates and parts available
    if response.candidates and response.candidates[0].content.parts:
        # Extract text from the first part of the first candidate
        generated_text = response.candidates[0].content.parts[0].text.strip()
        return generated_text
    else:
        return "No generated content available."

def generate_vertexai_newsresponse(newsprompt, data):
    formatted_data = f"""
    - News Summary: {data} 
    """
    full_prompt = newsprompt + formatted_data
    responses = model.generate_content(full_prompt)
    return extract_text_from_response(responses)