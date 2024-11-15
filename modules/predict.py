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
model = GenerativeModel("gemini-1.5-pro-002")
from prophet import Prophet
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def transform_price(df):
    """Transform the price data for Prophet model."""
    try:
        # Ensure we have a proper DataFrame
        if not isinstance(df, pd.DataFrame):
            logging.error(f"Input is not a DataFrame: {type(df)}")
            raise TypeError("Input must be a pandas DataFrame")
            
        # Log initial state
        logging.info(f"Initial DataFrame columns: {df.columns.tolist()}")
        logging.info(f"Initial DataFrame index: {df.index.name}")
        
        # Flatten the Adj Close column if it's 2D
        adj_close = df['Adj Close'].values.flatten() if isinstance(df['Adj Close'].values, np.ndarray) else df['Adj Close']
        
        # Create a clean DataFrame with Prophet's required columns (ds and y)
        if isinstance(df.index, pd.DatetimeIndex):
            temp_df = pd.DataFrame({
                'ds': df.index,
                'y': adj_close
            })
        else:
            if 'Date' in df.columns:
                temp_df = pd.DataFrame({
                    'ds': df['Date'],
                    'y': adj_close
                })
            else:
                raise KeyError("Neither DatetimeIndex nor 'Date' column found")
        
        # Ensure datetime type
        temp_df['ds'] = pd.to_datetime(temp_df['ds'])
        
        # Sort by date
        temp_df = temp_df.sort_values('ds').reset_index(drop=True)
        
        # Convert to float if necessary
        temp_df['y'] = temp_df['y'].astype(float)
        
        # Verify final structure
        logging.info(f"Final DataFrame columns: {temp_df.columns.tolist()}")
        logging.info(f"Final DataFrame shape: {temp_df.shape}")
        
        return temp_df
        
    except Exception as e:
        logging.error(f"Error in transform_price: {str(e)}", exc_info=True)
        raise

def train_prophet_model(df):
    """Train the Prophet model with the given data."""
    try:
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        # Check required columns
        required_cols = ['ds', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
            
        # Ensure data types
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Remove any NaN values
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data points after processing")
            
        logging.info(f"Training model with {len(df)} data points")
        
        # Initialize and train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # Controls flexibility of the trend
            seasonality_prior_scale=10.0,  # Controls flexibility of seasonality
            seasonality_mode='multiplicative'  # Better for stock prices
        )
        
        model.fit(df)
        return model
        
    except Exception as e:
        logging.error(f"Error in train_prophet_model: {str(e)}", exc_info=True)
        raise

def make_forecast(model, df, periods):
    """Make future predictions using the trained Prophet model."""
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Make predictions
    forecast = model.predict(future)
    
    # Add historical predictions
    historical_dates = df['ds']
    historical_predictions = forecast[forecast['ds'].isin(historical_dates)]
    
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