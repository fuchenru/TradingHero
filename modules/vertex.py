import streamlit as st
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import pandas as pd
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")

model = GenerativeModel("gemini-1.5-pro-002")

def generate_vertexai_response(prompt, symbol, symbol_prices, company_basic, news, recommendations):
    symbol_prices_str = symbol_prices.to_string()
    company_basic_str = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())
    formatted_data = f"""
    {prompt}\nSymbol: {symbol}\nPrices: {symbol_prices_str}\nBasics: {company_basic_str}\nNews: {news}\nRecommendations: {recommendations} 
    """
    full_prompt = prompt + formatted_data
    response = model.generate_content(full_prompt) 

    generated_text = response.text 

    return generated_text 