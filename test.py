from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.express as px
import streamlit as st
import yfinance as yf
import finnhub
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
import os
from datetime import date, datetime, timedelta
from collections import defaultdict
import streamlit as st
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots
from scipy import signal
import json
import scipy as sp
import warnings
import re
warnings.filterwarnings('ignore')
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
        # Fall back to Python 2's urllib2
    from urllib2 import urlopen
import certifi
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
# Define the model for Gemini Pro
model = GenerativeModel("gemini-1.5-pro-preview-0409")

# static_data.py
exchange_code_names = ['US exchanges (NYSE, Nasdaq)']

exchange_codes = ['US']

# data_retriever.py

def get_exchange_code_names():
    return exchange_code_names



def get_exchange_codes():
    return exchange_codes


def get_symbols(exchange_code):
    symbol_data = finnhub_client().stock_symbols(exchange_code)
    symbols = []
    for symbol_info in symbol_data:
        symbols.append(symbol_info['displaySymbol'])
    symbols.sort()
    return symbols


def today():
    from datetime import date, datetime, timedelta
    return date.today().strftime("%Y-%m-%d")


def n_weeks_before(date_string, n):

    date_value = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return date_value.strftime("%Y-%m-%d")


def n_days_before(date_string, n):
    from datetime import date, datetime, timedelta
    date_value = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=n)
    return date_value.strftime("%Y-%m-%d")


def get_current_stock_data(symbol, n_weeks):
    from datetime import date, datetime, timedelta
    today = date.today().strftime("%Y-%m-%d")
    # current_date = today()
    n_weeks_before_date = n_weeks_before(today, n_weeks)
    stock_data = yf.download(symbol, n_weeks_before_date, today)
    return stock_data



def finnhub_client():
    return finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")



def get_current_basics(symbol, day):
    basic_financials = finnhub_client().company_basic_financials(symbol, 'all')
    if not basic_financials['series']:
        return []

    basic_list, basic_dict = [], defaultdict(dict)

    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)

    basic_list.sort(key=lambda x: x['period'])

    for basic in basic_list[::-1]:
        if basic['period'] <= day:
            return basic

    return basic_list[-1]



def get_peers(symbol):
    return finnhub_client().company_peers(symbol)



def get_financials(symbol, freq):
    return finnhub_client().financials_reported(symbol=symbol, freq=freq)


def get_income_statement(symbol, freq='quarterly'):
    financials = get_financials(symbol, freq)
    financials_data = financials['data']
    dates = [financials_data['endDate'] for financials_data in financials_data]
    ic = [financials_data['report']['ic'] for financials_data in financials_data]
    return dates, ic


def get_revenue(symbol):
    return finnhub_client.stock_revenue_breakdown(symbol)


def get_basic_detail(df):
    metrics = pd.DataFrame.from_dict(df, orient='index', columns=['Value'])
    explanations = {
        "assetTurnoverTTM": "Asset Turnover (TTM): Measures how efficiently a company uses its assets to generate sales.",
        "bookValue": "Book Value per Share: Represents the net asset value per share of common stock.",
        "cashRatio": "Cash Ratio: Indicates a company's ability to cover its short-term liabilities with its cash and cash equivalents.",
        "currentRatio": "Current Ratio: Measures a company's ability to pay off its short-term liabilities with its current assets.",
        "ebitPerShare": "EBIT per Share: Earnings before interest and taxes per share, showing profitability from operations.",
        "eps": "Earnings per Share (EPS): Portion of a company's profit allocated to each outstanding share of common stock.",
        "ev": "Enterprise Value (EV): Represents the total value of a company, including its equity and debt.",
        "fcfMargin": "Free Cash Flow Margin: Percentage of revenue that converts into free cash flow.",
        "fcfPerShareTTM": "Free Cash Flow per Share (TTM): Free cash flow generated per share of common stock.",
        "grossMargin": "Gross Margin: Percentage of revenue remaining after deducting the cost of goods sold.",
        "inventoryTurnoverTTM": "Inventory Turnover (TTM): Measures how efficiently a company manages its inventory.",
        "longtermDebtTotalAsset": "Long-Term Debt to Total Assets: Portion of a company's assets financed by long-term debt.",
        "longtermDebtTotalCapital": "Long-Term Debt to Total Capital: Measures the proportion of a company's capital structure financed by long-term debt.",
        "longtermDebtTotalEquity": "Long-Term Debt to Total Equity: Indicates the financial risk of a company's capital structure.",
        "netDebtToTotalCapital": "Net Debt to Total Capital: Measures the proportion of a company's capital structure financed by net debt (total debt minus cash and equivalents).",
        "netDebtToTotalEquity": "Net Debt to Total Equity: Shows the financial risk considering both debt and cash.",
        "netMargin": "Net Margin: Percentage of revenue remaining after all expenses have been deducted.", 
        "operatingMargin": "Operating Margin: Percentage of revenue remaining after deducting operating costs.",
        "payoutRatioTTM": "Payout Ratio (TTM): Percentage of earnings paid out as dividends.",
        "pb": "Price-to-Book Ratio (P/B): Compares a company's market value to its book value.",
        "peTTM": "Price-to-Earnings Ratio (TTM): Shows how much investors are willing to pay per dollar of earnings.",
        "pfcfTTM": "Price-to-Free Cash Flow Ratio (TTM): Compares a company's market value to its free cash flow per share.",
        "pretaxMargin": "Pretax Margin: Percentage of revenue remaining after deducting all expenses except taxes.",
        "psTTM": "Price-to-Sales Ratio (TTM): Compares a company's market value to its revenue.",
        "quickRatio": "Quick Ratio: Measures a company's ability to meet short-term obligations with its most liquid assets.",
        "receivablesTurnoverTTM": "Receivables Turnover (TTM): Measures how efficiently a company collects its receivables.",
        "roaTTM": "Return on Assets (TTM): Indicates how effectively a company uses its assets to generate profit.",
        "roeTTM": "Return on Equity (TTM): Measures the profitability of a company in relation to its shareholders' equity.",
        "roicTTM": "Return on Invested Capital (TTM): Evaluates a company's efficiency at allocating its capital to profitable investments.",
        "rotcTTM": "Return on Total Capital (TTM): Measures the return generated on all forms of capital invested in the company.",
        "salesPerShare": "Sales per Share: Amount of revenue generated per share of common stock.",
        "sgaToSale": "SG&A to Sales: Ratio of selling, general, and administrative expenses to sales.", 
        "totalDebtToEquity": "Total Debt to Equity Ratio: Measures the financial risk and leverage of a company.",
        "totalDebtToTotalAsset": "Total Debt to Total Assets: Proportion of a company's assets financed by debt.",
        "totalDebtToTotalCapital": "Total Debt to Total Capital: Measures the proportion of a company's capital structure (debt and equity) that is financed by debt.",
        "totalRatio": "Total Ratio: Also known as the Debt-to-Asset Ratio, it indicates the proportion of a company's assets that are financed by debt.",
    }
    metrics['Explanation'] = metrics.index.map(explanations)
    return metrics

# bidder.py
def buy_and_hold(init_val, symbol_closes):
    shares = init_val/symbol_closes[0]
    trade_val = shares*symbol_closes[-1]
    return trade_val

def buy_rule(init_value, transact_dates, transact_percents, dates, symbol_closes):
    close_values = [symbol_closes[dates.index(transact_date)] for transact_date in transact_dates]
    wallet = init_value
    shares = 0.0
    for close_value, transact_percent in zip(close_values, transact_percents):
        is_buy = transact_percent > 0
        if is_buy:
            transact_val = wallet*transact_percent/100
            transact_shares = transact_val/close_value
            wallet -= transact_val
            shares += transact_shares

            # st.write(f"buy: R{transact_val} = {transact_shares}shares. Remaining: R{wallet} and {shares}shares")
        else:
            transact_shares = shares*-transact_percent/100
            transact_val = transact_shares*close_value
            shares -= transact_shares
            wallet += transact_val

            # st.write(f"sell: R{transact_val} = {transact_shares}shares. Remaining: R{wallet} and {shares}shares")

    share_value = shares*symbol_closes[-1]
    return wallet + share_value









# vertex.py
def extract_text_from_generation_response(responses):
        return [resp.text for resp in responses]



def generate_vertexai_response(prompt, symbol, symbol_prices, company_basic, news, recommendations):
    # Convert data to strings
    symbol_prices_str = symbol_prices.to_string()
    company_basic_str = "Keys: {}, Values: {}".format(company_basic.keys(), company_basic.values())

    # Create the full prompt
    full_prompt = f"{prompt}\nSymbol: {symbol}\nPrices: {symbol_prices_str}\nBasics: {company_basic_str}\nNews: {news}\nRecommendations: {recommendations}"

    responses = model.generate_content(
    [full_prompt],
    generation_config={
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 1
    },
        stream=True,
      )
  
    return extract_text_from_generation_response(responses)


# indicators.py
def price_at_index(index, dates, dataset):
    price = defaultdict(float)

    opens = dataset['Open']
    closes = dataset['Close']
    highs = dataset['High']
    lows = dataset['Low']

    price['Date'] = dates[index]
    price['Open'] = opens[index]
    price['Close'] = closes[index]
    price['High'] = highs[index]
    price['Low'] = lows[index]
    return price


def candle_is_green(current_price):
    return current_price['Open'] < current_price['Close']


def engulfing_candle(prev_price, current_price):
    prev_is_green = candle_is_green(prev_price)
    current_is_green = candle_is_green(current_price)
    from_negative_to_positive = not prev_is_green and current_is_green
    from_positive_to_negative = prev_is_green and not current_is_green
    is_bullish_engulfing = from_negative_to_positive and current_price['Close'] > prev_price['Open'] and current_price['Open'] < prev_price['Close']
    is_bearish_engulfing = from_positive_to_negative and current_price['Close'] < prev_price['Open'] and current_price['Open'] > prev_price['Close']
    return is_bullish_engulfing, is_bearish_engulfing


def engulfing_candle_bullish(prev_price, current_price):
    prev_is_green = candle_is_green(prev_price)
    current_is_green = candle_is_green(current_price)
    from_negative_to_positive = not prev_is_green and current_is_green
    is_bullish_engulfing = from_negative_to_positive and current_price['Close'] > prev_price['Open'] and current_price['Open'] < prev_price['Close']
    return is_bullish_engulfing


def engulfing_candle_bearish(prev_price, current_price):
    prev_is_green = candle_is_green(prev_price)
    current_is_green = candle_is_green(current_price)
    from_positive_to_negative = prev_is_green and not current_is_green
    is_bearish_engulfing = from_positive_to_negative and current_price['Close'] < prev_price['Open'] and current_price['Open'] > prev_price['Close']
    return is_bearish_engulfing


def create_engulfing_candle_bullish_indicators(dates, dataset):
    indicator = defaultdict(list)
    indicator_timestamps = indicator['Date']
    indicator_values = indicator['Values']
    
    prev_price = price_at_index(0, dates, dataset)
    for index in range(1, len(dates)):
        price = price_at_index(index, dates, dataset)
        is_engulfing = engulfing_candle_bullish(prev_price, price)
        if is_engulfing:
            indicator_timestamps.append(dates[index])
            offset = ((price['Close'] - price['Open']) - (prev_price['Open'] - prev_price['Close'])) * 5
            value = price['Close'] + offset
            indicator_values.append(value)
        prev_price = price

    indicator_dict = dict(indicator)
    indicator_dict['IsBullish'] = True

    return indicator_dict


def create_engulfing_candle_bearish_indicators(dates, dataset):
    indicator = defaultdict(list)
    indicator_timestamps = indicator['Date']
    indicator_values = indicator['Values']
    
    prev_price = price_at_index(0, dates, dataset)
    for index in range(1, len(dates)):
        price = price_at_index(index, dates, dataset)
        is_engulfing = engulfing_candle_bearish(prev_price, price)
        if is_engulfing:
            indicator_timestamps.append(dates[index])
            offset = ((price['Open'] - price['Close']) - (prev_price['Close'] - prev_price['Open'])) * 5
            value = price['Close'] - offset
            indicator_values.append(value)
        prev_price = price

    indicator_dict = dict(indicator)
    indicator_dict['IsBullish'] = False 

    return indicator_dict


def create_indicators(dates, dataset):
    indicators = defaultdict(dict)
    indicators['Engulfing Bullish'] = create_engulfing_candle_bullish_indicators(dates, dataset)
    indicators['Engulfing Bearish'] = create_engulfing_candle_bearish_indicators(dates, dataset)
    return indicators


# recommend.py
def get_rec(ticker_symbol):
    finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
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


# predict.py
def transform_price(df):
    df = df[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    return df

def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def calculate_performance_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def extract_text_from_generation_response(responses):
    """Extracts the concatenated text from the responses and removes extra newlines/spaces."""
    concatenated_text = []
    for candidate in responses.candidates:
        for part in candidate.content.parts:
            concatenated_text.append(part.text.strip())
    return concatenated_text[0]

def generate_vertexai_tsresponse(tsprompt, future_price,metrics_data):
    future_price = future_price.to_string()
    responses = model.generate_content([tsprompt, future_price, metrics_data])
    return extract_text_from_generation_response(responses)


# get_earnings.py
def get_earnings(ticker_symbol):
    finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
    earning = finnhub_client.company_earnings(ticker_symbol, limit=5)
    return earning

# status.py
def get_status(exchange_index):
    finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
    return finnhub_client.market_status(exchange=exchange_index)


def get_basic(ticker_symbol):
    finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
    return finnhub_client.company_basic_financials(ticker_symbol, 'all')["metric"]


# calculator.py

def linear_regression_line(dates, y_list):
    from datetime import date, datetime, timedelta
    if not any(dates):
        return 0, 0
    if isinstance(dates[0], datetime):
        dates = [ts.timestamp() for ts in dates]
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    mean_x = np.mean(dates)
    mean_y = np.mean(y_list)

    # Calculate the slope (m) and y-intercept (c) of the regression line
    m = np.sum((dates - mean_x) * (y_list - mean_y)) / np.sum((dates - mean_x) ** 2)
    c = mean_y - m * mean_x

    return m, c


def linear_regression_points(dates, y_list):
    from datetime import date, datetime, timedelta
    if not any(dates):
        return 0, 0
    if isinstance(dates[0], datetime):
        dates = [ts.timestamp() for ts in dates]
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    m, c = linear_regression_line(dates, y_list)

    return m * dates + c


def linear_regression(dates, y_list):
    if not any(dates):
        return 0, 0
    if isinstance(dates[0], datetime):
        dates = [ts.timestamp() for ts in dates]
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    m, c = linear_regression_line(dates, y_list)

    y_low = m * dates[0] + c
    y_high = m * dates[-1] + c

    return y_low, y_high


def normalize(data_list, high=1.0, low=-1.0):
    data = np.array(data_list)
    min_val = np.min(data)
    max_val = np.max(data)
    delta = max_val - min_val
    new_delta = high - low

    data = ((data - min_val) * new_delta / delta) + low
    return data

def intercepts(data1, data2):
    intercept_indices = []
    prev_data1_is_above = True
    for index, (data1_v, data2_v) in enumerate(zip(data1, data2)):
        data1_is_above = data1_v - data2_v >= 0.0
        if index is not 0 and (not any(intercept_indices) or index is not intercept_indices[-1] + 1) and prev_data1_is_above is not data1_is_above:
            intercept_indices.append(index)
        prev_data1_is_above = data1_is_above

    return intercept_indices

# ui.py

# chart datapoint icons
raw_symbols = SymbolValidator().values
up_arrow = raw_symbols[5]
down_arrow = raw_symbols[6]


def create_candlestick(fig, dates, dataset, title, y_label):
    candlestick = go.Candlestick(name=y_label, 
                                 x=dates,
                                 open=dataset['Open'],
                                 high=dataset['High'],
                                 low=dataset['Low'],
                                 close=dataset['Close'])
    fig.add_trace(candlestick)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )


def create_indicators(fig, datasets):
    for indicator in datasets:
        indicator_data = datasets[indicator]

        marker_color="lightskyblue"
        marker_symbol = 0
        if 'IsBullish' in indicator_data:
            if indicator_data['IsBullish']:
                marker_color = 'green'
                marker_symbol = 5
            else:
                marker_color = 'red'
                marker_symbol = 6

        indicator_plot = go.Scatter(name=indicator,
                                    mode="markers",
                                    x=indicator_data['Date'],
                                    y=indicator_data['Values'],
                                    marker_symbol=marker_symbol,
                                    marker_line_color="midnightblue",
                                    marker_color=marker_color,
                                    marker_line_width=2,
                                    marker_size=15,
                                    hovertemplate="%{indicator}: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>")
        fig.add_trace(indicator_plot)


def create_lines(fig, dates, datasets, title, y_label):
    for key in datasets:
        line = go.Scatter(name=key, x=dates, y=datasets[key])
        fig.add_trace(line)


def create_markers(fig, dates, dataset, title, y_label, marker_symbol=3, marker_color="blue", marker_size=15):
    line = go.Scatter(name=title, x=dates, y=dataset,
                      mode="markers",
                      marker_symbol=marker_symbol,
                      marker_line_color="midnightblue",
                      marker_color=marker_color,
                      marker_line_width=2,
                      marker_size=marker_size)
    fig.add_trace(line)


def create_line(fig, dates, dataset, title="title", y_label="values", marker_symbol=4, marker_size=15, color='rgba(0,100,80,0.2)'):
    line = go.Scatter(name=title, x=dates, y=dataset, marker_line_color="yellow", fillcolor=color)
    fig.add_trace(line)


def create_fill_area(fig, dates, y_low, y_high, title, color='rgba(0,100,80,0.2)'):
    # line_low = go.Scatter(name=title, x=dates, y=y_low, fillcolor=color, showlegend=False)
    # fig.add_trace(line_low)
    # line_high = go.Scatter(name=title, x=dates, y=y_low, fillcolor=color, showlegend=False)
    # fig.add_trace(line_high)
    fill_area = go.Scatter(
        name=title,
        x=dates + dates[::-1],
        y=y_high + y_low[::-1],
        fill='toself',
        fillcolor=color,
        line=dict(color=color)
    )
    fig.add_trace(fill_area)


def create_spectrogram(dates, data_list, sampling_frequency=1, num_points_fft=128, overlap_percent=50.0, title="title", log_scale=True):
    data_list = normalize(data_list, 1, -1)

    # Spectrogram
    w = signal.blackman(num_points_fft)
    freqs, bins, pxx = signal.spectrogram(data_list, sampling_frequency, window=w, nfft=num_points_fft, noverlap=int(num_points_fft*overlap_percent/100.0))

    dates_subset = [dates[int(bin)] for bin in bins]

    if log_scale:
        z = 10 * np.log10(pxx)
    else:
        z = pxx

    trace = [go.Heatmap(
        x=dates_subset,
        y=freqs,
        z=z,
        colorscale='Jet',
    )]
    layout = go.Layout(
        title=title,
        yaxis=dict(title='Frequency'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    fig = go.Figure(data=trace, layout=layout)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

    return fig


def create_heatmap(dates, data_list, bin_count=10, time_steps=20, title='title'):
    time_width = int(len(dates)/time_steps)
    dates_subset = [dates[index*time_width] for index in range(time_steps)]
    min_val = np.min(data_list)
    max_val = np.max(data_list)
    delta = (max_val-min_val)
    min_val -= 0.2*delta
    max_val += 0.2*delta
    delta = (max_val-min_val)

    bin_width = delta/(bin_count + 1)
    bins = np.arange(min_val, max_val, bin_width)

    values = np.empty(shape=(time_steps, bin_count))
    for time_index in range(time_steps):
        data_subset = data_list[time_index*time_width:time_index*time_width+time_width]
        counts, res_bins = np.histogram(data_subset, bins=bins)
        values[time_index:] = counts

    trace = [go.Heatmap(
        x=dates_subset,
        y=bins,
        z=values.transpose(),
        colorscale='Jet',
    )]
    layout = go.Layout(
        title=title,
        yaxis=dict(title='Values'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    fig = go.Figure(data=trace, layout=layout)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

    return fig


def add_mouse_indicator(fig, selected_points, min, max):
    if any(selected_points):
        fig.add_shape(
                go.layout.Shape(
                    type='line',
                    x0=selected_points[0]['x'],
                    x1=selected_points[0]['x'],
                    y0=min,
                    y1=max,
                    line=dict(color='red', width=2),
                )
            )


# trends.py
# plot the trend of the market as a candlestick graph.
def sector_trends(symbol, weeks_back):
    # 1. get all peers
    peers = get_peers(symbol)
    # 2. get data for each peer
    peers_stock_data = defaultdict(list)
    dates = []
    for peer in peers:
        peer_data = get_current_stock_data(peer, weeks_back)
        if len(peer_data) == 0:
            continue
        if not any(dates):
            dates = peer_data.index.format()
        peers_stock_data[peer] = peer_data

    # 3. normalize all data (get min->max value, then set each value to (X-min)/(max-min)
    peers_stock_data_normalized = defaultdict(dict)
    indicator_stock_data_normalized_peer_sum = defaultdict(list)
    for peer, peer_stock_data in peers_stock_data.items():
        peer_stock_data_normalized = defaultdict(list)
        for indicator in peer_stock_data:
            indicator_normalized = peer_stock_data_normalized[indicator]
            indicator_stock_data_normalized_sum = indicator_stock_data_normalized_peer_sum[indicator]

            indicator_values = peer_stock_data[indicator]
            min_val = np.min(indicator_values)
            max_val = np.max(indicator_values)
            delta = max_val - min_val

            value_idx = 0
            for value in indicator_values:
                normalized = (value - min_val)/delta
                indicator_normalized.append(normalized)

                while len(indicator_stock_data_normalized_sum) <= value_idx:
                    indicator_stock_data_normalized_sum.append(0)
                indicator_stock_data_normalized_sum[value_idx] += normalized
                value_idx += 1

        peers_stock_data_normalized[peer] = peer_stock_data_normalized
    # 4. get the average value for each indicator [open, close, high, low] for each time-step.
    peer_count = len(peers)
    indicator_stock_data_normalized_peer_avg = defaultdict(list)
    for indicator, indicator_sum_values in indicator_stock_data_normalized_peer_sum.items():
        indicator_avg_values = indicator_stock_data_normalized_peer_avg[indicator]
        for sum_value in indicator_sum_values:
            indicator_avg_values.append(sum_value/peer_count)
    # 5. plot the resulting normalized-averaged-indicators in a candlestick chart.  
    close_data = defaultdict(list)
    for peer, peer_data_normalized in peers_stock_data_normalized.items():
        close_data[peer] = peer_data_normalized['Close']
    
    relative_close_data = defaultdict(list)
    for peer in peers_stock_data_normalized:
        relative_close_data[peer] = [a - b for a, b in zip(close_data[peer], indicator_stock_data_normalized_peer_avg['Close'])]

    return dates, close_data, relative_close_data, indicator_stock_data_normalized_peer_avg


def trend_line(date_indices, data_list, min_trend_size=7, step_size=1, sigma_multiplier=1):
    trend_dates, trend_values = [], []
    if not any(date_indices) or not any(data_list):
        return trend_dates, trend_values

    np_dates = np.array([ts.timestamp() for ts in date_indices])
    dates = date_indices.format()

    start_index = 0
    index = min_trend_size

    while index < len(data_list):
        np_dates_subset = np_dates[start_index:index]
        np_values_subset = data_list[start_index:index]
        # for the value range, calculate linear_regression and standard deviation
        m, c = linear_regression_line(np_dates_subset, np_values_subset)
        predicted_points = m * np_dates_subset + c
        relative_mean = np.mean(np.abs(predicted_points - np_values_subset))  # TODO: use Welford's algorithm

        # for the next datapoint(s), calculate the next value in that trend,
        # and check if it is n*trend_sigma away from the current trend line
        next_index = index + step_size
        if next_index >= len(data_list):
            break

        x = np_dates[next_index:next_index+1]
        y = data_list[next_index:next_index+1]
        expected_y = m*x + c
        dev = np.mean(np.abs(y - expected_y))

        if dev > relative_mean * sigma_multiplier:
            # store the current date and value as a trend changer
            trend_dates.append(dates[index])
            trend_values.append(data_list[index])
            # reset the calculation for the next data
            start_index = next_index - min_trend_size

        index = next_index

    return trend_dates, trend_values


def vwap(symbol_price_data):
    typical_price = (symbol_price_data['High'] + symbol_price_data['Low'] + symbol_price_data['Close']) / 3.0
    volume = symbol_price_data['Volume']
    vwap_values = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap_values


def sma(prices, period=20):
    values = []
    for index in range(len(prices) - period):
        values.append(np.mean(prices[index:index + period]))
    return values


def bollinger_bands(dates, symbol_price_data, n_days_deviation=20, deviation_multiplier=2):
    bollinger_dates, bollinger_high, bollinger_low = [], [], []
    closes = symbol_price_data['Close']

    for index in range(len(closes) - n_days_deviation):
        bollinger_dates.append(dates[index + n_days_deviation])

        closes_subset = closes[index:index + n_days_deviation]
        sma = np.sum(closes_subset)/len(closes_subset)
        sigma = deviation_multiplier * np.sqrt(np.sum(np.power(closes_subset - sma, 2))/len(closes_subset))
        bollinger_high.append(sma + sigma)
        bollinger_low.append(sma - sigma)
    return bollinger_dates, bollinger_low, bollinger_high


def support_lines():
    a = 1


# a linear regression over a period
def linear_regression_trend(prices, period=32):
    np_dates = np.array([ts.timestamp() for ts in prices.index])
    np_values = np.array(prices)

    value_count = len(prices)
    trend = []
    for index in range(value_count-period):
        np_dates_subset = np_dates[index:index+period+1]
        np_values_subset = np_values[index:index+period+1]
        low, high = linear_regression(np_dates_subset, np_values_subset)
        trend.append(high)

    return trend


def bids(dates, symbol_prices, period=14):
    close_values = symbol_prices['Close']
    trend = linear_regression_trend(close_values, period=period)
    sma_values = sma(close_values, period=period)

    # get intercepts
    dates_subset = dates[period:]
    close_subset = close_values[period:]

    sma_intercept_indices = intercepts(sma_values, close_subset)
    trend_intercept_indices = intercepts(trend, close_subset)

    sma_intercept_dates = [dates_subset[index] for index in sma_intercept_indices]
    bids = []
    for index in sma_intercept_indices:
        is_up = sma_values[index] > close_subset[index]
        if is_up:
            bids.append(10)
        else:
            bids.append(-10)

    return sma_intercept_dates, bids

# get_news.py

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
from datetime import date, datetime, timedelta
today = date.today()
prev_date = today - timedelta(weeks=3)

formatted_today = today.strftime('%Y-%m-%d')
formatted_prevdate = prev_date.strftime('%Y-%m-%d')

def classify_sentiment(text):
    result = nlp(text)
    
    label = result[0]['label']
    
    if label == 'negative':
        return 'Negative 😡'
    elif label == 'neutral':
        return 'Neutral 😐'
    elif label == 'positive':
        return 'Positive 🙂'
    else:
        return 'Error: Unknown label'

def prediction_score(text):
    result = nlp(text)
    score = result[0]['score'] * 100
    return(f"{score:.2f}%")


def count_related_words(text, ticker_symbol):
    # Find all occurrences of the AI-related keywords in the text
    matches = re.findall(ticker_symbol, text, flags=re.IGNORECASE)
    # Return the number of matches
    return len(matches)

def get_stock_news(ticker_symbol):
    finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=formatted_prevdate, to=formatted_today)
        df = pd.DataFrame(news)
        df = df[['headline', 'summary']]
        df = df[~df['summary'].str.contains('Looking for stock market analysis and research with proves results?')]
    
        # Select words less than 500 and reset index by counting related word
        df['word_counts'] = df['summary'].str.split().str.len()
        df['related_word_count'] = df['summary'].apply(lambda text: count_related_words(text, ticker_symbol))

        df = df[(df['word_counts'] <= 500)]

        df.rename(columns={'headline': 'Headline', 'summary': 'Summary'}, inplace=True)
        df['Sentiment Analysis'] = df['Summary'].apply(classify_sentiment)
        df['Score'] = df['Summary'].apply(prediction_score)
        df = df.drop(columns=['word_counts', 'related_word_count'])
        return df.head(5)
    except Exception as e:
        return f"An error occurred: {e}"


def get_active_symbols():
    exchange_names = get_exchange_code_names()
    exchange_name = exchange_names[0] if len(exchange_names) == 1 else st.session_state.exchange_name
    exchange_index = exchange_names.index(exchange_name)
    exchange = get_exchange_codes()[exchange_index]
    return get_symbols(exchange)

st.set_page_config(layout="wide", page_title='TradingHero', page_icon="https://i.imgur.com/Lw9T6s9.png")
st.markdown(
        """
        <style>
        .css-1d391kg {width: 359px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
st.sidebar.title("Menu 🌐")
st.image("https://i.imgur.com/WQE6iLY.jpeg", width=805)

# Use session_state to track which section was last opened
if 'last_opened' not in st.session_state:
    st.session_state['last_opened'] = None

with st.sidebar:
    if st.sidebar.button('ℹ️ Overall Information'):
        st.session_state['last_opened'] = 'Overall Information'
    if st.sidebar.button('📜 Historical Stock and EPS Surprises'):
        st.session_state['last_opened'] = 'End-of-Day Historical Stock and EPS Surprises'
    if st.sidebar.button('💡Stock Analyst Recommendations and Latest News'):
        st.session_state['last_opened'] = 'Stock Analyst Recommendations'
    if st.sidebar.button('🔍 Trends Forecasting and TradingHero Analysis'):
        st.session_state['last_opened'] = 'Trends'

vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
model = GenerativeModel("gemini-1.5-pro-preview-0409")

    # Define the generation configuration
generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    # Define safety settings
safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

if st.session_state['last_opened'] == 'Overall Information':
    col1, col2 = st.columns(2)
    with col1:
        exchange_names = get_exchange_code_names()
        # Right now only limit to 1 stock(US Exchange)
        if len(exchange_names) == 1:
            exchange_name = exchange_names[0]  
            st.write("**Exchange (Currently only support US market):**")
            # st.text("Exchange (Currently only support US market):")
            st.markdown(f"**{exchange_name}**")
        else:
            # If there are multiple exchange options, let the user select
            exchanges_selectbox = st.selectbox(
                'Exchange (Currently only support US market):',
                exchange_names,
                index=exchange_names.index('US exchanges (NYSE, Nasdaq)'),
                key='exchange_selectbox'
            )
            exchange_name = exchanges_selectbox  # Use the user's selection

        exchange_index = exchange_names.index(exchange_name)
        exchange = get_exchange_codes()[exchange_index]

        symbols = get_active_symbols()
        selected_symbol = st.selectbox(
            'Select Stock Ticker:',
            symbols,
            index=symbols.index(st.session_state.get('selected_symbol', 'AAPL')),
            key='overall_info_symbol'
        )

    st.session_state['selected_symbol'] = selected_symbol
    symbol = st.session_state.get('selected_symbol', symbols[0])

    market_status = get_status(exchange)
    st.write("**Market Status Summary:**")
    if market_status["isOpen"]:
        st.write("🟢 The US market is currently open for trading.")
    else:
        st.write("🔴 The US market is currently closed.")
        if market_status["holiday"]:
            st.write(f"Reason: {market_status['holiday']}")
        else:
            st.write("🌃 It is currently outside of regular trading hours.")

    company_basic = get_basic(symbol)
    if st.checkbox('Show Company Basics'):
        basics_data = get_current_basics(symbol, today())
        metrics = get_basic_detail(basics_data)
        st.dataframe(metrics[['Explanation', 'Value']], width=3000)

    # with col2:
    st.text_input('No. of years look-back:', value=1, key="years_back")
    years_back = int(st.session_state.years_back)
    weeks_back = years_back * 13 * 4

    symbol_prices = get_current_stock_data(symbol, weeks_back)
    #     if not symbol_prices.empty:
    dates = symbol_prices.index.astype(str)
    #         st.text_input('No. of days back-test:', value=0, key="backtest_period")
    #         n_days_back = int(st.session_state.backtest_period)
    #         end_date = n_days_before(today(), n_days_back)
    #         symbol_prices_backtest = symbol_prices[symbol_prices.index <= end_date]
    #         backtest_dates = symbol_prices_backtest.index.astype(str)
    candleFigure = make_subplots(rows=1, cols=1)
    create_candlestick(candleFigure, dates, symbol_prices, symbol, 'Price')

    # plot all
    candleFigure.update_layout(title="Candle Chart",
                                xaxis_title='Date',
                                yaxis_title="Price per Share",
                                template='plotly_dark')

    # use this to add markers on other graphs for click points on this graph
    selected_points = []


    st.plotly_chart(candleFigure, use_container_width=True)
    st.write("**Trading Hero AI Technical Summary:**")
    prices = symbol_prices.loc[:,"Adj Close"]
    symbol = st.session_state.get('selected_symbol', symbols[0])
    text1 = f"""You are a financial analyst tasked with providing a technical summary for various stocks based on their recent price movements and technical indicators. Your analysis should include an evaluation of the stock\'s trend, its performance relative to a major market index (like the S&P 500), and key technical indicators such as momentum (measured by the RSI), volume trends, and the position relative to moving averages.

        Please generate a technical summary that follows the structure and tone of the example provided below:

        Example Technical Summary:
        \"Although the stock has pulled back from higher prices, [Ticker] remains susceptible to further declines. A reversal of the existing trend looks unlikely at this time. Over the last 50 trading days, when compared to the S&P 500, the stock has performed in line with the market. Despite a weak technical condition, there are positive signs. Momentum, as measured by the 9-day RSI, is bullish. Over the last 50 trading sessions, there has been more volume on down days than on up days, indicating that [Ticker] is under distribution, which is a bearish condition. The stock is currently above a falling 50-day moving average. A move below this average could trigger additional weakness in the stock. [Ticker] could find secondary support at its rising 200-day moving average.\"

        Ticker Symbol: {symbol}
        Current Price: {prices}
        [Detailed Technical Summary]"""

        # Generate content based on the prompt
    responses = model.generate_content(
            [text1],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

    # responses = model.generate_content(
    # [text1],
    # generation_config=generation_config,
    # safety_settings=safety_settings,
    # stream=True,)

    def extract_text_from_generation_response(responses):
        return [resp.text for resp in responses] 

    # Extract and format the response
    text_responses = extract_text_from_generation_response(responses)
    full_summary = "".join(text_responses)  # Join all parts into one string
    st.write(full_summary)  # Display the formatted summary

if st.session_state['last_opened'] == 'End-of-Day Historical Stock and EPS Surprises':
    # Get symbols and exchange data as with the candle chart
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])

    # Retrieve stock data for the selected ticker
    if 'symbol_prices_hist' not in st.session_state or st.session_state.selected_symbol_hist != symbol:
        st.session_state.symbol_prices_hist = get_current_stock_data(symbol, 52 * 5)  # 5 years of weekly data
        st.session_state.selected_symbol_hist = symbol

    if not st.session_state.symbol_prices_hist.empty:
        symbol_prices_hist = st.session_state.symbol_prices_hist[::-1]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**End-of-Day Historical Stock Data for {symbol}**")
            st.dataframe(symbol_prices_hist, width=1000)

        with col2:
            st.markdown(f"**Historical EPS Surprises for {symbol}**")
            earnings_data = get_earnings(symbol)
            if earnings_data:
                # Reverse the data for chronological order (oldest to newest)
                earnings_data = earnings_data[::-1]
                actuals = [item['actual'] for item in earnings_data]
                estimates = [item['estimate'] for item in earnings_data]
                periods = [item['period'] for item in earnings_data]
                surprisePercents = [item['surprisePercent'] for item in earnings_data]
                surpriseText = ['Beat: {:.2f}'.format(item['surprise']) if item['surprise'] > 0 else 'Missed: {:.2f}'.format(item['surprise']) for item in earnings_data]

                # Create the bubble chart for EPS surprises
                fig = go.Figure()

                # Add actual EPS values with marker sizes based on surprise
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=actuals,
                    mode='markers+text',
                    name='Actual',
                    text=surpriseText,
                    textposition="bottom center",
                    marker=dict(
                        size=[abs(s) * 10 + 5 for s in surprisePercents],  # Dynamically size markers
                        color='LightSkyBlue',
                        line=dict(
                            color='MediumPurple',
                            width=2
                        ),
                        sizemode='diameter'
                    )
                ))

                # Add estimated EPS values as smaller, semi-transparent markers
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=estimates,
                    mode='markers',
                    name='Estimate',
                    marker=dict(
                        size=30,  # Fixed size for all estimate markers
                        color='MediumPurple',
                        opacity=0.5,
                        line=dict(
                            color='MediumPurple',
                            width=2
                        )
                    )
                ))

                # Customize the layout
                fig.update_layout(
                    title='Historical EPS Surprises',
                    xaxis_title='Period',
                    yaxis_title='Quarterly EPS',
                    xaxis=dict(
                        type='category',
                        tickmode='array',
                        tickvals=periods,
                        ticktext=periods
                    ),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig)
            else:
                st.error("No EPS data available for the selected symbol.")
    else:
        st.error("No historical stock data available for the selected symbol.")


if st.session_state['last_opened'] == 'Stock Analyst Recommendations':
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Stock Analyst Recommendations for {symbol}**")
    
    if symbol:
        recommendations = get_rec(symbol)
        if recommendations:
            st.subheader('Stock Analyst Recommendations')
            st.bar_chart(recommendations)
        else:
            st.error("No recommendations data available for the selected symbol.")
    
    
    with st.spinner("NLP sentiment analysis is working to generate."):
        progress_bar = st.progress(0)
        if st.checkbox('Show Latest News'):
            st.write("""
                Trading Hero utilizes a self-trained Natural Language Processing (NLP) pipeline to perform sentiment analysis 
                specifically tailored for financial news. This self-trained NLP solution leverages state-of-the-art machine 
                learning models with 98% accuracy to interpret and classify the sentiment of textual data from news articles 
                related to stock market activities.
            """)
            news_data = get_stock_news(symbol)
            if not news_data.empty:
                news_data.set_index("Headline", inplace=True)
                progress_bar.progress(50)
                st.table(news_data)
                progress_bar.progress(100)
            else:
                st.error("No news data available for the selected symbol.")
                progress_bar.progress(100)

if st.session_state['last_opened'] == 'Trends':
    
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Trends and Forecast for {symbol}**")

    period = st.slider(label='Select Period for Trend Analysis (Days)', min_value=7, max_value=140, value=14, step=7)
    days_to_forecast = st.slider('Days to Forecast:', min_value=30, max_value=365, value=90)
    years_back = st.number_input('No. of years look-back:', value=1, min_value=1, max_value=10)
    weeks_back = int(years_back * 52)

    symbol_prices = get_current_stock_data(symbol, weeks_back)

    if symbol_prices.empty:
        st.error("No data available for the selected symbol.")
        # return

    dates = symbol_prices.index.format()

    # Create the indicator figure for trend analysis
    indicator_figure = make_subplots(rows=1, cols=1)
    create_line(indicator_figure, dates, symbol_prices['Close'], "Close Price", color="red")

    # Volume Weighted Average Price (VWAP)
    vwap = vwap(symbol_prices)
    create_line(indicator_figure, dates, vwap, "Volume Weighted Average Price (VWAP)", "VWAP")

    # Bollinger Bands
    bollinger_dates, bollinger_low, bollinger_high = bollinger_bands(dates, symbol_prices)
    create_fill_area(indicator_figure, bollinger_dates, bollinger_low, bollinger_high, "Bollinger Bands")

    # Linear Regression Trend
    trend = linear_regression_trend(symbol_prices['Close'], period=period)
    create_line(indicator_figure, dates[period:], trend, f"Trend: {period} Day", color='blue')

    # Trend Deviation Area
    price_deviation_from_trend = [price - trend_val for price, trend_val in zip(symbol_prices['Close'][period:], trend)]
    deviation_price_offset = [price + dev for price, dev in zip(symbol_prices['Close'][period:], price_deviation_from_trend)]
    create_fill_area(indicator_figure, dates[period:], trend, deviation_price_offset, f"Trend Deviation: {period} Day Trend", color="rgba(100,0,100,0.3)")

    # Simple Moving Average (SMA)
    deltas_sma = sma(symbol_prices['Close'], period)
    create_line(indicator_figure, dates[period:], deltas_sma, f"SMA {period} Day", color='green')

    # Bids and Strategies
    init_val = 1000.0
    buy_and_hold_val = buy_and_hold(init_val, symbol_prices['Close'])
    st.write(f"Initial Value: {init_val}")
    st.write(f"BUY AND HOLD: {buy_and_hold_val}")

    bid_dates, bid_vals = bids(dates, symbol_prices, period)
    strat_val = buy_rule(init_val, bid_dates, bid_vals, dates, symbol_prices['Close'])

    create_markers(indicator_figure, bid_dates, bid_vals, "Bids", "Price")
    st.write(f"STRATEGY: {strat_val}")

    indicator_figure.update_layout(
        title="Indicators",
        xaxis_title='Date',
        yaxis_title="Price",
        template='plotly_dark'
    )
    st.plotly_chart(indicator_figure, use_container_width=True)

    st.markdown(
        f"Trend deviation: indicates the extent to which the price is deviating from its current trajectory. "
        f"The trajectory is the linear regression of the price over {period} days.")
    st.markdown('''
        <ol>
            <li>Mark all intercepts between the SMA and the price.</li>
            <li>Calculate the cumulative sum of trend deviations. The cumulative sum is reset when an intercept occurs.</li>
            <li>When there is an intercept and the cumulative sum is above a threshold, then a price reversal is imminent.</li>
        </ol>
    ''', unsafe_allow_html=True)

    st.markdown(
            f"If the trend intersects SMA, then reset. "
            f"When the trend intersects the price, place a bid. If the previous intercept is lower, then buy.")

    # Forecast Section
    st.subheader('Trading Hero Forecasting')
    st.write("""
        The plot below visualizes the forecasted stock prices using Trading Hero's own time-series algorithm.
        Our tool is designed to handle the complexities of time series data automatically, such as seasonal variations and missing data.
        The plotted forecast includes trend lines and confidence intervals, providing a clear visual representation of expected future values and the uncertainty around these predictions.
    """)
    df = transform_price(symbol_prices)
    tsmodel = train_prophet_model(df)
    forecast = make_forecast(tsmodel, days_to_forecast)

    fig1 = plot_plotly(tsmodel, forecast)
    fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
    st.plotly_chart(fig1)

    actual = df['y']
    predicted = forecast['yhat'][:len(df)]
    metrics = calculate_performance_metrics(actual, predicted)
    st.subheader('Performance Metrics')

    metrics_data = {
        "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
        "Value": [metrics['MAE'], metrics['MSE'], metrics['RMSE']]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Metric", inplace=True)
    st.table(metrics_df)
    # AI analysis
    future_price = forecast.loc[:,"trend"]
    metrics_data = "Keys: {}, Values: {}".format(metrics.keys(), metrics.values())
    tsprompt = f"""
    You are provided with the following data for one company's future stock:
    {metrics_data}
    - Performance Metrics:
    - MAE: 
    - MSE: 
    - RMSE: 

    Based on this information, please provide insights into the company's potential investment implications.
    """

    # Generate content based on the prompt
    responses = model.generate_content(
            [tsprompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

    responses = model.generate_content(
    [tsprompt],
    generation_config=generation_config,
    safety_settings=safety_settings,
    stream=True,)

    def extract_text_from_generation_response(responses):
        return [resp.text for resp in responses] 

    # Extract and format the response
    text_responses = extract_text_from_generation_response(responses)
    full_summary = "".join(text_responses)  # Join all parts into one string
    # st.write(full_summary)  # Display the formatted summary
    if st.button("Show Trading Hero Time-Series AI Analysis"):
        st.markdown(full_summary)

    recommendations = get_rec(symbol)
    if st.button("Show Trading Hero AI Analysis"):
        
        company_basic = get_basic(symbol)
        # company_basic = get_current_basics(symbol, today())
        prices = symbol_prices.loc[:,"Adj Close"]
        news_data = get_stock_news(symbol)["Summary"].to_string()
        analyst_rec = "Keys: {}, Values: {}".format(recommendations.keys(), recommendations.values())
        # ai_data = generate_vertexai_response(input_prompt,symbol,prices,company_basic,news_data,analyst_rec)
        # for response in ai_data:
        #     st.markdown(response)
        input_prompt = f"""
        As a seasoned market analyst with an uncanny ability to decipher the language of price charts, your expertise is crucial in navigating the turbulent seas of financial markets. I have provided you with information about a specific stock, including its ticker symbol, recent prices, company fundamental information, news, and analyst recommendations. Your task is to analyze the stock and provide insights on its recent performance and future prospects.

        The first few characters you received is the company's ticker symbol{symbol}. 

        Analysis Guidelines:
        1. Company Overview: Begin with a brief overview of the company you are analyzing. Understand its market position, recent news, financial health, and sector performance to provide context for your analysis.

        2. Fundamental Analysis {company_basic}: Conduct a thorough fundamental analysis of the company. Assess its financial statements, including income statements, balance sheets, and cash flow statements. Evaluate key financial ratios (e.g., P/E ratio, debt-to-equity, ROE) and consider the company's growth prospects, management effectiveness, competitive positioning, and market conditions. This step is crucial for understanding the underlying value and potential of the company.

        3. Pattern Recognition {prices}: Diligently examine the price chart to identify critical candlestick formations, trendlines, and a comprehensive set of technical indicators relevant to the timeframe and instrument in question. Pay special attention to recent price movements in the year 2024.

        4. Technical Analysis: Leverage your in-depth knowledge of technical analysis principles to interpret the identified patterns and indicators. Extract nuanced insights into market dynamics, identify key levels of support and resistance, and gauge potential price movements in the near future.

        5. Sentiment Prediction {news_data}: Based on your technical analysis, predict the likely direction of the stock price. Determine whether the stock is poised for a bullish upswing or a bearish downturn. Assess the likelihood of a breakout versus a consolidation phase, taking into account the analyst recommendations.

        6. Confidence Level: Evaluate the robustness and reliability of your prediction. Assign a confidence level based on the coherence and convergence of the technical evidence at hand.

        Put more weight on the Pattern Recognition and the news.{news_data}

        Finally, provide your recommendations on whether to Buy, Hold, Sell, Strong Buy, or Strong Sell the stock in the future, along with the percentage of confidence you have in your prediction.
        """
        responses = model.generate_content(
        [input_prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,)

    # responses = model.generate_content(
    # [text1],
    # generation_config=generation_config,
    # safety_settings=safety_settings,
    # stream=True,)

    def extract_text_from_generation_response(responses):
        return [resp.text for resp in responses] 

    # Extract and format the response
    text_responses = extract_text_from_generation_response(responses)
    full_summary = "".join(text_responses)  # Join all parts into one string
    st.write(full_summary)  # Display the formatted summary
        
