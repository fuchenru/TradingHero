from collections import defaultdict

import numpy as np
import scipy as sp

import calculator
import data_retriever

import streamlit as st


# plot the trend of the market as a candlestick graph.
def sector_trends(symbol, weeks_back):
    # 1. get all peers
    peers = data_retriever.get_peers(symbol)
    # 2. get data for each peer
    peers_stock_data = defaultdict(list)
    dates = []
    for peer in peers:
        peer_data = data_retriever.get_current_stock_data(peer, weeks_back)
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
        m, c = calculator.linear_regression_line(np_dates_subset, np_values_subset)
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
        low, high = calculator.linear_regression(np_dates_subset, np_values_subset)
        trend.append(high)

    return trend


def bids(dates, symbol_prices, period=14):
    close_values = symbol_prices['Close']
    trend = linear_regression_trend(close_values, period=period)
    sma_values = sma(close_values, period=period)

    # get intercepts
    dates_subset = dates[period:]
    close_subset = close_values[period:]

    sma_intercept_indices = calculator.intercepts(sma_values, close_subset)
    trend_intercept_indices = calculator.intercepts(trend, close_subset)

    sma_intercept_dates = [dates_subset[index] for index in sma_intercept_indices]
    bids = []
    for index in sma_intercept_indices:
        is_up = sma_values[index] > close_subset[index]
        if is_up:
            bids.append(10)
        else:
            bids.append(-10)

    return sma_intercept_dates, bids

