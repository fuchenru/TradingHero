import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
import google.generativeai as genai
genai.configure(api_key="AIzaSyDfwWZH59XOxiVnz6XRyXWIX47hF7BZ1jQ")
model = genai.GenerativeModel(model_name="gemini-pro")

def transform_price(df):
    df = df[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df

def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def calculate_performance_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def generate_gemini_tsresponse(tsprompt, future_price,metrics_data):
    future_price = future_price.to_string()
    response = model.generate_content([tsprompt, future_price, metrics_data])
    return response.text