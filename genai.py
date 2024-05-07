import streamlit as st
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models


vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.0-pro")



def extract_text_from_generation_response(responses):
    """Extracts the concatenated text from the responses and removes extra newlines/spaces."""
    concatenated_text = []
    for response in responses:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                concatenated_text.append(part.text.strip())
    return concatenated_text



def generate_vertexai_response(prompt, symbol, symbol_prices, company_basic, news, recommendations):
    # Convert data to strings
    symbol_prices_str = symbol_prices.to_string()
    company_basic_str = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())

    # Create the full prompt
    full_prompt = f"{prompt}\nSymbol: {symbol}\nPrices: {symbol_prices_str}\nBasics: {company_basic_str}\nNews: {news}\nRecommendations: {recommendations}"

    responses = model.generate_content(
    [full_prompt],
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.9,
        "top_p": 1
    },
        stream=True,
      )
  
    return extract_text_from_generation_response(responses)

# def generate_chatgpt_response(messages, symbol, symbol_prices, company_basic,news,recommendations):
#     openai.api_key = "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"
#     symbol_prices = symbol_prices.to_string()
#     company_basic = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())
#     messages = [
#         {"role": "system", "content": "As a professional stock market analyst, your expertise is crucial in navigating the turbulent seas of financial markets. I have provided you with information about a specific stock, including its ticker symbol, recent prices, company fundamental information, news, and analyst recommendations. Your task is to analyze the stock and provide insights on its recent performance and future prospects.The first few characters you received is the company's ticker symbol. "},
#         {"role": "user", "content": "Analysis Guidelines: 1. Company Overview: Begin with a brief overview of the company you are analyzing. Understand its market position, recent news, financial health, and sector performance to provide context for your analysis.2. Fundamental Analysis: Conduct a thorough fundamental analysis of the company. Assess its financial statements, including income statements, balance sheets, and cash flow statements. Evaluate key financial ratios (e.g., P/E ratio, debt-to-equity, ROE) and consider the company's growth prospects, management effectiveness, competitive positioning, and market conditions. This step is crucial for understanding the underlying value and potential of the company. 3. Pattern Recognition: Diligently examine the price chart to identify critical candlestick formations, trendlines, and a comprehensive set of technical indicators relevant to the timeframe and instrument in question. Pay special attention to recent price movements in the year 2024. 4. Technical Analysis: Leverage your in-depth knowledge of technical analysis principles to interpret the identified patterns and indicators. Extract nuanced insights into market dynamics, identify key levels of support and resistance, and gauge potential price movements in the near future. 5. Sentiment Prediction: Based on your technical analysis, predict the likely direction of the stock price. Determine whether the stock is poised for a bullish upswing or a bearish downturn. Assess the likelihood of a breakout versus a consolidation phase, taking into account the analyst recommendations. 6. Confidence Level: Evaluate the robustness and reliability of your prediction. Assign a confidence level based on the coherence and convergence of the technical evidence at hand. Put more weight on the Pattern Recognition and the news. Finally, provide your recommendations on whether to Buy, Hold, Sell, Strong Buy, or Strong Sell the stock in the future, along with the percentage of confidence you have in your prediction."},
#     ]

#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages
#     )

#     return completion.choices[0].message['content']
