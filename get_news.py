import finnhub
from datetime import date, timedelta
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models

vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
model = GenerativeModel("gemini-1.5-flash-preview-0514")
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
tokenizer = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("fuchenru/Trading-Hero-LLM")
nlp = pipeline("text-classification", model=sentiment_model, tokenizer=tokenizer)

def preprocess(text, tokenizer, max_length=128):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return inputs

def predict_sentiment(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: 'Neutral 😐', 1: 'Positive 🙂', 2: 'Negative 😡'}
    predicted_sentiment = label_map[predicted_label]
    return predicted_sentiment

def get_stock_news(ticker_symbol, start_date, end_date):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
        if df.empty:
            return pd.DataFrame()
        top_5_news = df.head(7)
        top_5_news['Sentiment Analysis'] = top_5_news['summary'].apply(predict_sentiment)
        return top_5_news
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

def get_all_stock_news(ticker_symbol, start_date, end_date):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
        if df.empty:
            return pd.Series()
        return df['headline']
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return pd.Series()

def extract_text_from_response(response):
    if response.candidates and response.candidates[0].content.parts:
        generated_text = response.candidates[0].content.parts[0].text.strip()
        return generated_text
    else:
        return "No generated content available."

def generate_vertexai_newsresponse(newsprompt, data):
    formatted_data = f"- News Summary: {data}"
    full_prompt = newsprompt + formatted_data
    responses = model.generate_content(full_prompt)
    return extract_text_from_response(responses)