import finnhub
import streamlit as st
import plotly.graph_objs as go
import json

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")


def get_earnings(ticker_symbol):
    earning = finnhub_client.company_earnings(ticker_symbol, limit=5)
    return earning