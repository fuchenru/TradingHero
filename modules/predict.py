import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models


vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-pro-002")

def transform_price(df):
    """Transform the price data for Prophet model."""
    df = df[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df

def train_prophet_model(df):
    """Train the Prophet model with the given data, with error handling for Cloud Run."""
    model = Prophet()
    try:
        model.fit(df)
    except RuntimeError as e:
        print(f"Error during Prophet optimization: {e}")
        # Try using a different optimization algorithm if newton fails
        model = Prophet()
        model.fit(df, method='lbfgs')
    return model

def make_forecast(model, periods):
    """Make future predictions using the trained Prophet model."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def calculate_performance_metrics(actual, predicted):
    """Calculate performance metrics for model evaluation."""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def extract_text_from_generation_response(responses):
    """Extract the concatenated text from the responses and removes extra newlines/spaces."""
    concatenated_text = []
    for candidate in responses.candidates:
        for part in candidate.content.parts:
            concatenated_text.append(part.text.strip())
    return concatenated_text[0]

def generate_vertexai_tsresponse(tsprompt, future_price, metrics_data):
    """Generate AI response using Vertex AI."""
    future_price = future_price.to_string()
    responses = model.generate_content([tsprompt, future_price, metrics_data])
    return extract_text_from_generation_response(responses)