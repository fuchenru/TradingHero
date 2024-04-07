import yfinance as yf
import finnhub
import pandas as pd
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

@st.cache_data
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