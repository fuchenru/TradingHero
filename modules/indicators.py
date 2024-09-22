# Defines candlestick indicators, and creates data-points for a dataset when an indicator is detected.

import streamlit as st
from collections import defaultdict


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

