import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
import io

genai.configure(api_key="AIzaSyDfwWZH59XOxiVnz6XRyXWIX47hF7BZ1jQ")
model = genai.GenerativeModel(model_name="gemini-pro")


def generate_gemini_response(prompt, symbol, symbol_prices, company_basic):
    symbol_prices = symbol_prices.to_string()
    basic = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())
    response = model.generate_content([prompt, symbol, symbol_prices, basic])
    return response.text