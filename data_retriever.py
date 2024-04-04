import yfinance as yf
import finnhub

import os
from datetime import date, datetime, timedelta
from collections import defaultdict

import static_data

#test
import streamlit as st


@st.cache_data
def get_exchange_code_names():
    return static_data.exchange_code_names


@st.cache_data
def get_exchange_codes():
    return static_data.exchange_codes


@st.cache_data
def get_symbols(exchange_code):
    symbol_data = finnhub_client().stock_symbols(exchange_code)
    symbols = []
    for symbol_info in symbol_data:
        symbols.append(symbol_info['displaySymbol'])
    symbols.sort()
    return symbols


@st.cache_data
def today():
    return date.today().strftime("%Y-%m-%d")


def n_weeks_before(date_string, n):
    date_value = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return date_value.strftime("%Y-%m-%d")


def n_days_before(date_string, n):
    date_value = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=n)
    return date_value.strftime("%Y-%m-%d")

@st.cache_data
def get_current_stock_data(symbol, n_weeks):
    current_date = today()
    n_weeks_before_date = n_weeks_before(current_date, n_weeks)
    stock_data = yf.download(symbol, n_weeks_before_date, current_date)
    return stock_data


@st.cache_data
def finnhub_client():
    return finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")


@st.cache_data
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


@st.cache_data
def get_peers(symbol):
    return finnhub_client().company_peers(symbol)


@st.cache_data
def get_financials(symbol, freq):
    return finnhub_client().financials_reported(symbol=symbol, freq=freq)


@st.cache_data
def get_income_statement(symbol, freq='quarterly'):
    financials = get_financials(symbol, freq)
    financials_data = financials['data']
    dates = [financials_data['endDate'] for financials_data in financials_data]
    ic = [financials_data['report']['ic'] for financials_data in financials_data]
    return dates, ic

@st.cache_data
def get_revenue(symbol):
    return finnhub_client.stock_revenue_breakdown(symbol)