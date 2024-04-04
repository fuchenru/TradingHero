import finnhub
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

def get_stock_news(ticker_symbol):
    return finnhub_client.company_news(ticker_symbol, _from="2024-04-01", to="2024-04-30")[0:5]