import finnhub
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")

def get_status(exchange_index):
    return finnhub_client.market_status(exchange=exchange_index)


def get_basic(ticker_symbol):
    return finnhub_client.company_basic_financials(ticker_symbol, 'all')["metric"]