import streamlit as st
from pathlib import Path
import google.generativeai as genai

# Assuming genai is a package you have access to and it's correctly installed
genai.configure(api_key="AIzaSyACWIyIUYRVS7k0wwJzzRQV4vBLnae01IQ")
model = genai.GenerativeModel(model_name="gemini-pro-vision")

def read_image_data(image_file):
    if image_file is None:
        raise FileNotFoundError("No image uploaded")
    return {"mime_type": image_file.type, "data": image_file.getvalue()}

def generate_gemini_response(prompt, image_file):
    image_data = read_image_data(image_file)
    response = model.generate_content([prompt, image_data])
    return response.text

st.title("Image Analysis with Gemini")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    input_prompt = """
    As a seasoned market analyst with an uncanny ability to decipher the language of price charts, your expertise is crucial in navigating the turbulent seas of financial markets. You will be presented with static images of historical stock charts, where your keen eye will dissect the intricate dance of candlesticks, trendlines, and technical indicators. Armed with this visual intelligence, you will unlock the secrets hidden within these graphs, predicting the future trajectory of the depicted stock with remarkable accuracy.
    """
    
    try:
        response = generate_gemini_response(input_prompt, uploaded_file)
        st.markdown(response)
    except Exception as e:
        st.error(f"Failed to generate response: {e}")

