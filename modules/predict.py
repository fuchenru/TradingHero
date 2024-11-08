import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models

# Import NeuralProphet
from neuralprophet import NeuralProphet, set_random_seed

vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-flash-002")

def transform_price(df):
    """Transform the price data for NeuralProphet model."""
    df = df[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def train_neuralprophet_model(df):
    """Train the NeuralProphet model with the given data."""
    np_model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        trend_reg=1.0,      # Increased trend regularization
        n_changepoints=5)   # Reduced number of changepoints
    np_model.fit(df, freq='D', progress='off')
    return np_model

def make_forecast(model, df, periods):
    """Make future predictions using the trained NeuralProphet model."""
    future = model.make_future_dataframe(df=df, periods=periods, n_historic_predictions=True)
    forecast = model.predict(future)
    return forecast

def calculate_performance_metrics(actual, predicted):
    """Calculate performance metrics for model evaluation."""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def extract_text_from_generation_response(responses):
    """Extract the concatenated text from the responses and remove extra newlines/spaces."""
    concatenated_text = []
    for candidate in responses.candidates:
        for part in candidate.content.parts:
            concatenated_text.append(part.text.strip())
    return concatenated_text[0]

def generate_vertexai_tsresponse(tsprompt, future_price, metrics_data):
    """Generate AI response using Vertex AI."""
    future_price_str = future_price.to_string(index=False)
    responses = model.generate_content([tsprompt, future_price_str, metrics_data])
    return extract_text_from_generation_response(responses)