import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly, plot
from datetime import datetime

import calculator
import data_retriever
import trends
import ui
import bidder
import get_news
import status
import recommend
import genai
import predict
import get_earnings

input_prompt = """
As a seasoned market analyst with an uncanny ability to decipher the language of price charts, your expertise is crucial in navigating the turbulent seas of financial markets. I have provided you with information about a specific stock, including its ticker symbol, recent prices, company fundamental information, news, and analyst recommendations. Your task is to analyze the stock and provide insights on its recent performance and future prospects.

The first few characters you received is the company's ticker symbol. 

Analysis Guidelines:
1. Company Overview: Begin with a brief overview of the company you are analyzing. Understand its market position, recent news, financial health, and sector performance to provide context for your analysis.

2. Fundamental Analysis: Conduct a thorough fundamental analysis of the company. Assess its financial statements, including income statements, balance sheets, and cash flow statements. Evaluate key financial ratios (e.g., P/E ratio, debt-to-equity, ROE) and consider the company's growth prospects, management effectiveness, competitive positioning, and market conditions. This step is crucial for understanding the underlying value and potential of the company.

3. Pattern Recognition: Diligently examine the price chart to identify critical candlestick formations, trendlines, and a comprehensive set of technical indicators relevant to the timeframe and instrument in question. Pay special attention to recent price movements in the year 2024.

4. Technical Analysis: Leverage your in-depth knowledge of technical analysis principles to interpret the identified patterns and indicators. Extract nuanced insights into market dynamics, identify key levels of support and resistance, and gauge potential price movements in the near future.

5. Sentiment Prediction: Based on your technical analysis, predict the likely direction of the stock price. Determine whether the stock is poised for a bullish upswing or a bearish downturn. Assess the likelihood of a breakout versus a consolidation phase, taking into account the analyst recommendations.

6. Confidence Level: Evaluate the robustness and reliability of your prediction. Assign a confidence level based on the coherence and convergence of the technical evidence at hand.

Put more weight on the Pattern Recognition and the news.

Finally, provide your recommendations on whether to Buy, Hold, Sell, Strong Buy, or Strong Sell the stock in the future, along with the percentage of confidence you have in your prediction.
"""


def run():
    # st.title("Trading Hero Stock Analysis")

    image_url = "https://i.imgur.com/WQE6iLY.jpeg"
    st.image(image_url, width=700)

    col1, col2 = st.columns(2)
    with col1:
        exchange_names = data_retriever.get_exchange_code_names()
        # Right now only limit to 1 stock(US Exchange)
        if len(exchange_names) == 1:
            exchange_name = exchange_names[0]  
            st.text("Exchange (Currently only support US market):")
            st.markdown(f"**{exchange_name}**")
        else:
            # If there are multiple exchange options, let the user select
            exchanges_selectbox = st.selectbox(
                'Exchange (Currently only support US market):',
                exchange_names,
                index=exchange_names.index('US exchanges (NYSE, Nasdaq)')
            )
            exchange_name = exchanges_selectbox  # Use the user's selection
    
        exchange_index = exchange_names.index(exchange_name)
        exchange = data_retriever.get_exchange_codes()[exchange_index]

        symbols = data_retriever.get_symbols(exchange)
        symbols_selectbox = st.selectbox(
            'Stock Ticker:',
            symbols,
            index=symbols.index('AAPL')
        )
        symbol = symbols_selectbox
    
    market_status = status.get_status(exchange)
    st.write("**Market Status Summary:**")
    if market_status["isOpen"]:
        st.write("🟢 The US market is currently open for trading.")
    else:
        st.write("🔴 The US market is currently closed.")
        if market_status["holiday"]:
            st.write(f"Reason: {market_status['holiday']}")
        else:
            st.write("🌃 It is currently outside of regular trading hours.")

    company_basic = status.get_basic(symbol)
    if st.checkbox('Company Basics'):
        basics_data = data_retriever.get_current_basics(symbol, data_retriever.today())
        metrics = data_retriever.get_basic_detail(basics_data)
        st.dataframe(metrics[['Explanation', 'Value']], width=3000)
        
    with col2:
        # max time period
        st.text_input('No. of years look-back:', value=1, key="years_back")
        years_back = int(st.session_state.years_back)
        weeks_back = years_back * 13 * 4

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
    candleFigure.update_layout(title="Candle Chart",
                               xaxis_title='Date',
                               yaxis_title="Price per Share",
                               template='plotly_dark')

    # use this to add markers on other graphs for click points on this graph
    selected_points = []


    st.plotly_chart(candleFigure, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(symbol_prices, width=1000)
    
    with col4:
        if True:
            data = get_earnings.get_earnings(symbol)
            data = data[::-1] 
            actuals = [item['actual'] for item in data]
            estimates = [item['estimate'] for item in data]
            periods = [item['period'] for item in data]
            surprisePercents = [item['surprisePercent'] for item in data]
            surpriseText = ['Beat: {:.2f}'.format(item['surprise']) if item['surprise'] > 0 else 'Missed: {:.2f}'.format(item['surprise']) for item in data]

            # Create the bubble chart
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
                    size=30,
                    #size=[abs(s) * 10 for s in surprisePercents], # Scale factor for visualization
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
                    size=30, # Fixed size for all estimate markers
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
                title= 'Historical EPS Surprises',
                xaxis_title='Period',
                yaxis_title='Quarterly EPS',
                xaxis=dict(
                    title='Period',
                    type='category', # Setting x-axis to category to display periods exactly
                    tickmode='array',
                    tickvals=periods,
                    ticktext=periods
                ),
                yaxis=dict(showgrid=False),
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig)  

    st.write('Stock Analyst Recommendations')
    if symbol:
        recommendations = recommend.get_rec(symbol)
        st.bar_chart(recommendations)
        
    st.write('Latest News')
    if True:
        news_data = get_news.get_stock_news(symbol)
        news_data.set_index("headline", inplace = True)
        st.table(news_data)

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
    
    if st.checkbox('Sector Trends'):
        # plot the trend of the market as a candlestick graph.
        fig2 = make_subplots(rows=1, cols=1)
        dates, close_data, relative_close_data, sector_normalized_avg = trends.sector_trends(symbol, weeks_back)
        ui.create_candlestick(fig2, dates, sector_normalized_avg, 'Sector Trend', 'Normalized Price')
        # st.plotly_chart(fig2, use_container_width=True)

        # plot the difference between each peer and the sector trend
        fig3 = make_subplots(rows=1, cols=1)
        ui.create_lines(fig3, dates, relative_close_data, "Peer 'Close' Relative to Sector", 'Relative Close')
        st.plotly_chart(fig3, use_container_width=True)

    forecast_days = int(st.session_state.years_back) * 365
    # if st.checkbox('Forecast Stock Prices'):
    if True:
        with st.spinner('Fetching data...'):
            df = predict.transform_price(symbol_prices)

        with st.spinner('Training model...'):
            model = predict.train_prophet_model(df)
            forecast = predict.make_forecast(model, forecast_days)

        st.subheader('Trading Hero Forecasting')
        st.write("""The plot below visualizes the forecasted stock prices using Trading Hero own time-series algorithm. 
                 Our tool is designed to handle the complexities of time series data automatically, 
                 such as seasonal variations and missing data. The plotted forecast includes trend lines and confidence intervals, 
                 providing a clear visual representation of expected future values and the uncertainty around these predictions. """)

        fig1 = plot_plotly(model, forecast)
        fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
        st.plotly_chart(fig1)

        actual = df['y']
        predicted = forecast['yhat'][:len(df)]
        metrics = predict.calculate_performance_metrics(actual, predicted)
        st.subheader('Performance Metrics')
        # st.write('The metrics below provide a quantitative measure of the model’s accuracy. They include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), with lower values indicating better performance.')

        metrics_data = {
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
            "Value": [metrics['MAE'], metrics['MSE'], metrics['RMSE']]
        }

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index("Metric", inplace = True)
        st.table(metrics_df)

        future_price = forecast.loc[:,"trend"]
        metrics_data = "Keys: {}, Values: {}".format(metrics.keys(), metrics.values())
        tsprompt = """
        You are provided with the following data for one company's stock:
        - Future Price Trend: {}
        - Performance Metrics:
        - MAE: {:.2f}
        - MSE: {:.2f}
        - RMSE: {:.2f}

        Based on this information, please provide insights into the company's future stock price trajectory and potential investment implications.
        """
        tsai_data = predict.generate_gemini_tsresponse(tsprompt,future_price,metrics_data)

        progress_bar = st.progress(0)
        if st.button("Show Trading Hero Time-Series AI Analysis"):
            progress_bar.progress(50)
            st.markdown(tsai_data)
            progress_bar.progress(100)

    with st.spinner("Model is working to generate."):
        progress_bar = st.progress(0)
        if st.button("Show Trading Hero AI Analysis"):
            company_basic = data_retriever.get_current_basics(symbol, data_retriever.today())
            prices = symbol_prices.loc[:,"Adj Close"]
            news_data = get_news.get_stock_news(symbol)["summary"].to_string()
            analyst_rec = "Keys: {}, Values: {}".format(recommendations.keys(), recommendations.values())
            ai_data = genai.generate_gemini_response(input_prompt,symbol,prices,company_basic,news_data,analyst_rec)
            progress_bar.progress(50)
            st.markdown(ai_data)
            progress_bar.progress(100)