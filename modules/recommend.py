import finnhub
import pandas as pd
import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-flash-002")
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

def get_rec(ticker_symbol: str) -> dict:
    try:
        data = finnhub_client.recommendation_trends(ticker_symbol)
        if not data:
            return {}

        df = pd.DataFrame(data)
        recommendations = {
            "Buy": df['buy'].sum(),
            "Hold": df['hold'].sum(),
            "Sell": df['sell'].sum(),
            "Strong Buy": df['strongBuy'].sum(),
            "Strong Sell": df['strongSell'].sum(),
        }
        return recommendations
    except Exception as e:
        st.error(f"Error fetching recommendations: {e}")
        return {}
    

def extract_text_from_generation_response(responses) -> str:
    concatenated_text = []
    for candidate in responses.candidates:
        for part in candidate.content.parts:
            concatenated_text.append(part.text.strip())
    return concatenated_text[0] if concatenated_text else ""

def generate_vertexai_recommendresponse(tsprompt: str, data: dict) -> str:
    formatted_data = f"""
    - Buy: {data.get('Buy', 0)} 
    - Hold: {data.get('Hold', 0)} 
    - Sell: {data.get('Sell', 0)} 
    - Strong Buy: {data.get('Strong Buy', 0)} 
    - Strong Sell: {data.get('Strong Sell', 0)} 
    """
    full_prompt = tsprompt + formatted_data
    responses = model.generate_content(full_prompt)
    return extract_text_from_generation_response(responses)
