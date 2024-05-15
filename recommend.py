import finnhub
import pandas as pd
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-flash-preview-0514")
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

def get_rec(ticker_symbol):
    data = finnhub_client.recommendation_trends(ticker_symbol)
    
    df = pd.DataFrame(data)
    
    recommendations = {
        "Buy": df['buy'].sum(),
        "Hold": df['hold'].sum(),
        "Sell": df['sell'].sum(),
        "Strong Buy": df['strongBuy'].sum(),
        "Strong Sell": df['strongSell'].sum(),
    }
    
    return recommendations

def extract_text_from_generation_response(responses):
    concatenated_text = []
    for candidate in responses.candidates:
        for part in candidate.content.parts:
            concatenated_text.append(part.text.strip())
    return concatenated_text[0]

def generate_vertexai_recommendresponse(tsprompt, data):
    formatted_data = f"""
    - Buy: {data['Buy']} 
    - Hold: {data['Hold']} 
    - Sell: {data['Sell']} 
    - Strong Buy: {data['Strong Buy']} 
    - Strong Sell: {data['Strong Sell']} 
    """
    full_prompt = tsprompt + formatted_data
    responses = model.generate_content(full_prompt)
    return extract_text_from_generation_response(responses)
