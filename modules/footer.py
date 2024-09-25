import streamlit as st

def add_footer():
    st.markdown("""
    ---
    © 2024 Trading Hero. All rights reserved.
                    
    **Disclaimer:** The stock ratings, signals, and rankings provided by Trading Hero are not intended to be investment advice. 
    These ratings, signals, and rankings are based on Artificial Intelligence (AI) analysis, which calculates probabilities, 
    not certainties. All the information contained on this website is for research and educational purposes and will never 
    be considered a recommendation or advice on investment, nor will it be considered legal, tax, or any other type of advice.
    """, unsafe_allow_html=True)