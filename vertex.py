import streamlit as st
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import pandas as pd
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-flash-preview-0514")



# def extract_text_from_generation_response(responses):
#     """Extracts the concatenated text from the responses and removes extra newlines/spaces."""
#     concatenated_text = []
#     for response in responses:
#         for candidate in response.candidates:
#             for part in candidate.content.parts:
#                 concatenated_text.append(part.text.strip())
#     return concatenated_text

def extract_text_from_generation_response(responses):
        return [resp.text for resp in responses]



def generate_vertexai_response(prompt, symbol, symbol_prices, company_basic, news, recommendations):
    # Convert data to strings
    symbol_prices_str = symbol_prices.to_string()
    company_basic_str = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())

    # Create the full prompt
    full_prompt = f"{prompt}\nSymbol: {symbol}\nPrices: {symbol_prices_str}\nBasics: {company_basic_str}\nNews: {news}\nRecommendations: {recommendations}"

    responses = model.generate_content(
    [full_prompt],
    generation_config={
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 1
    },
        stream=True,
      )
  
    return extract_text_from_generation_response(responses)