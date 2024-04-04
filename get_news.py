import yfinance as yf

def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    news = stock.get_news()
    return news