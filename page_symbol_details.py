import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go

import calculator
import data_retriever
import trends
import ui
import bidder
import get_news
import status
import recommend

def run():
    col1, col2 = st.columns(2)
    with col1:
        exchange_names = data_retriever.get_exchange_code_names()
        exchanges_selectbox = st.selectbox(
            'Exchange:',
            exchange_names,
            index=exchange_names.index('US exchanges (NYSE, Nasdaq)')
        )
        exchange_name = exchanges_selectbox
        exchange_index = exchange_names.index(exchange_name)
        exchange = data_retriever.get_exchange_codes()[exchange_index]

        symbols = data_retriever.get_symbols(exchange)
        symbols_selectbox = st.selectbox(
            'Stock:',
            symbols,
            index=symbols.index('AAPL')
        )
        symbol = symbols_selectbox
    
    market_status = status.get_status(exchange)
    st.json(market_status)

    company_basic = status.get_basic(symbol)
    st.json(company_basic)

    with col2:
        # max time period
        st.text_input('No. of years look-back:', value=1, key="years_back")
        years_back = int(st.session_state.years_back)
        weeks_back = years_back * 12 * 4

        symbol_prices = data_retriever.get_current_stock_data(symbol, weeks_back)
        if not any(symbol_prices):
            return
        dates = symbol_prices.index.format()

        # back test
        st.text_input('No. of days back-test:', value=0, key="backtest_period")
        n_days_back = int(st.session_state.backtest_period)
        end_date = data_retriever.n_days_before(data_retriever.today(), n_days_back)
        symbol_prices_backtest = symbol_prices[symbol_prices.index <= end_date]
        backtest_dates = symbol_prices_backtest.index.format()

    # symbol candlestick graph
    candleFigure = make_subplots(rows=1, cols=1)
    ui.create_candlestick(candleFigure, dates, symbol_prices, symbol, 'Price')

    # plot all
    candleFigure.update_layout(title="Symbol Ticker",
                               xaxis_title='Date',
                               yaxis_title="Price per Share",
                               template='plotly_dark')

    # use this to add markers on other graphs for click points on this graph
    selected_points = []


    st.plotly_chart(candleFigure, use_container_width=True)

    if st.checkbox('Trends'):
        period = st.slider(label='period', min_value=7, max_value=140, value=14, step=7)

        indicatorFigure = make_subplots(rows=1, cols=1)
        ui.create_line(indicatorFigure, dates, symbol_prices['Close'], "Close price", color="red")

        # vwap
        vwap = trends.vwap(symbol_prices_backtest)
        ui.create_line(candleFigure, backtest_dates, vwap, "Volume Weighted Average Price (VWAP)", "VWAP")

        # bollinger bands
        bollinger_dates, bollinger_low, bollinger_high = trends.bollinger_bands(backtest_dates, symbol_prices_backtest)
        ui.create_fill_area(candleFigure, bollinger_dates, bollinger_low, bollinger_high, "Bollinger Bands")

        # trend
        trend = trends.linear_regression_trend(symbol_prices['Close'][:], period=period)
        ui.create_line(indicatorFigure, dates[period:], trend, f"Trend: {period} day", color='blue')

        # trend deviation area
        price_deviation_from_trend = [price - trend_val for price, trend_val in
                                      zip(symbol_prices['Close'][period:], trend)]
        deviation_price_offset = [price + dev for price, dev in
                                  zip(symbol_prices['Close'][period:], price_deviation_from_trend)]
        ui.create_fill_area(indicatorFigure, dates[period:], trend, deviation_price_offset,
                            f"Trend Deviation: {period} day trend", color="rgba(100,0,100,0.3)")

        # sma line
        deltas_sma = trends.sma(symbol_prices['Close'], period)
        ui.create_line(indicatorFigure, dates[period:], deltas_sma, f"SMA {period} day", color='green')

        # bids
        init_val = 1000.0
        buy_and_hold_val = bidder.buy_and_hold(init_val, symbol_prices['Close'])
        st.write(f"BUY AND HOLD: {buy_and_hold_val}")

        bid_dates, bid_vals = trends.bids(dates, symbol_prices, period)
        strat_val = bidder.buy_rule(init_val, bid_dates, bid_vals, dates, symbol_prices['Close'])

        ui.create_markers(indicatorFigure, bid_dates, bid_vals, "Bids", "Price")
        st.write(f"STRAT: {strat_val}")

        indicatorFigure.update_layout(title="Indicators",
                                   xaxis_title='Date',
                                   yaxis_title="Price",
                                   template='plotly_dark')
        st.plotly_chart(indicatorFigure, use_container_width=True)

        # doxy
        st.markdown(
            f"Trend deviation: indicates the extent by which the price is deviating from its current trajectory. "
            f"The trajectory is the linear regression of the price over {period} days.")
        st.markdown('''<ol><li>mark all intercepts between the SMA and the price.</li>
                    <li>calculate the cum-sum of trend deviations. the cum-sum is reset when an intercept occurs.</li>
                    <li>when there is an intercept, and the cum-sum above a threshold, then a price reversal is imminent.</li></ol>''', unsafe_allow_html=True)

        st.markdown('''<ol><li>if the trend intersects SMA, then reset.</li>
                        <li>when the trend intersects the price, place a bid. if the previous intercept is lower, then buy.</li></ol>''')

    # if st.checkbox('Tests'):
    #     deltas = [cur - symbol_prices['Close'][max(0, idx - 1)] for idx, cur in enumerate(symbol_prices['Close'])]
    #     testFigure2 = make_subplots(rows=1, cols=1)
    #     ui.create_line(testFigure2, dates, deltas, "Deltas")
    #     ui.add_mouse_indicator(testFigure2, selected_points, min=np.min(deltas),
    #                            max=np.max(deltas))
    #     testFigure2.update_layout(title="Deltas")

    #     ui.create_heatmap(dates, deltas, title="Deltas heatmap")

    #     # sma
    #     sma_dates, deltas_sma = trends.sma(dates, deltas)
    #     ui.create_line(testFigure2, sma_dates, deltas_sma, "Deltas SMA")
    #     test_id = st.plotly_chart(testFigure2, use_container_width=True, key='test2')

    if st.checkbox('Sector Trends'):
        # plot the trend of the market as a candlestick graph.
        fig2 = make_subplots(rows=1, cols=1)
        dates, close_data, relative_close_data, sector_normalized_avg = trends.sector_trends(symbol, weeks_back)
        ui.create_candlestick(fig2, dates, sector_normalized_avg, 'Sector Trend', 'Normalized Price')
        st.plotly_chart(fig2, use_container_width=True)

        # plot the difference between each peer and the sector trend
        fig3 = make_subplots(rows=1, cols=1)
        ui.create_lines(fig3, dates, relative_close_data, "Peer 'Close' Relative to Sector", 'Relative Close')
        st.plotly_chart(fig3, use_container_width=True)

    if st.checkbox('Symbol Basics'):
        basics_data = data_retriever.get_current_basics(symbol, data_retriever.today())
        st.write(basics_data)
    
    if st.checkbox('Latest News'):
        basics_data = get_news.get_stock_news(symbol)
        st.write(basics_data)

    st.title("Stock Analyst Recommendations")
    if symbol:
        recommendations = recommend.get_rec(symbol)
        st.bar_chart(recommendations)