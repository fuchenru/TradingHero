import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
import io
import openai
genai.configure(api_key="AIzaSyDfwWZH59XOxiVnz6XRyXWIX47hF7BZ1jQ")
model = genai.GenerativeModel(model_name="gemini-pro")


def generate_gemini_response(prompt, symbol, symbol_prices, company_basic,news,recommendations):
    symbol_prices = symbol_prices.to_string()
    company_basic = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())
    response = model.generate_content([prompt, symbol, symbol_prices, company_basic, news,recommendations])
    return response.text

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