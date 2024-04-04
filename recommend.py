import finnhub
import pandas as pd

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