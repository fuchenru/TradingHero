import finnhub
from datetime import date

finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

today = date.today()
formatted_today = today.strftime('%Y-%m-%d')

def get_stock_news(ticker_symbol):
    return finnhub_client.company_news(ticker_symbol, _from=formatted_today, to=formatted_today)[0:5]
