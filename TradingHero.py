import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime

# Your imports (truncated for brevity)
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

def get_active_symbols():
    exchange_names = data_retriever.get_exchange_code_names()
    exchange_name = exchange_names[0] if len(exchange_names) == 1 else st.session_state.exchange_name
    exchange_index = exchange_names.index(exchange_name)
    exchange = data_retriever.get_exchange_codes()[exchange_index]
    return data_retriever.get_symbols(exchange)

def run():
    # Custom CSS to set the sidebar width
    st.markdown(
        """
        <style>
        .css-1d391kg {width: 350px;}
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
        if st.sidebar.button('□ ℹ️ Overall Information'):
            st.session_state['last_opened'] = 'Overall Information'
        if st.sidebar.button('□ 📈 Candle Chart'):
            st.session_state['last_opened'] = 'Candle Chart'
        if st.sidebar.button('□ 📜 End-of-Day Historical Stock and EPS Surprises'):
            st.session_state['last_opened'] = 'End-of-Day Historical Stock and EPS Surprises'
        if st.sidebar.button('□ 💡Stock Analyst Recommendations'):
            st.session_state['last_opened'] = 'Stock Analyst Recommendations'
        if st.sidebar.button('□ 📰 Latest News'):
            st.session_state['last_opened'] = 'Latest News'
        if st.sidebar.button('□ 🔍 Trends'):
            st.session_state['last_opened'] = 'Trends'
        if st.sidebar.button('□ 📈Forecast and Performance Metrics'):
            st.session_state['last_opened'] = 'Forecast and Performance Metrics'
        if st.sidebar.button('□ 🤖TradingHero AI Analysis'):
            st.session_state['last_opened'] = 'TradingHero AI Analysis'



    # Check which content to display based on what was clicked in the sidebar
    if st.session_state['last_opened'] == 'Overall Information':
        show_overall_information()
    elif st.session_state['last_opened'] == 'Candle Chart':
        show_candle_chart()
    elif st.session_state['last_opened'] == 'End-of-Day Historical Stock and EPS Surprises':
        show_historical_data()
    elif st.session_state['last_opened'] == 'Stock Analyst Recommendations':
        show_analyst_recommendations()
    elif st.session_state['last_opened'] == 'Latest News':
        show_latest_news()
    elif st.session_state['last_opened'] == 'Trends':
        show_trends()
    elif st.session_state['last_opened'] == 'Forecast and Performance Metrics':
        show_forecast_metrics()
    elif st.session_state['last_opened'] == 'TradingHero AI Analysis':
        show_ai_analysis()


def show_overall_information():
    col1, col2 = st.columns(2)
    with col1:
        exchange_names = data_retriever.get_exchange_code_names()
        if len(exchange_names) == 1:
            exchange_name = exchange_names[0]  
            st.text("Exchange (Currently only support US market):")
            st.markdown(f"**{exchange_name}**")
        else:
            exchanges_selectbox = st.selectbox(
                'Exchange (Currently only support US market):',
                exchange_names,
                index=exchange_names.index('US exchanges (NYSE, Nasdaq)'),
                key='exchange_selectbox'
            )
            exchange_name = exchanges_selectbox
    
        exchange_index = exchange_names.index(exchange_name)
        exchange = data_retriever.get_exchange_codes()[exchange_index]
        symbols = get_active_symbols()
    
        # Let the user select a stock and store it in session_state
        selected_symbol = st.selectbox(
            'Select Stock Ticker:',
            symbols,
            index=symbols.index(st.session_state.get('selected_symbol', 'AAPL')),
            key='overall_info_symbol'
    )

    # Save the selected symbol into session_state so that other functions can access it
    st.session_state['selected_symbol'] = selected_symbol
    symbol = st.session_state.get('selected_symbol', symbols[0])
    
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
    if st.checkbox('Show Company Basics'):
        basics_data = data_retriever.get_current_basics(symbol, data_retriever.today())
        metrics = data_retriever.get_basic_detail(basics_data)
        st.dataframe(metrics[['Explanation', 'Value']], width=3000)

    with col2:
        st.text_input('No. of years look-back:', value=1, key="years_back")
        years_back = int(st.session_state.years_back)
        weeks_back = years_back * 13 * 4

        symbol_prices = data_retriever.get_current_stock_data(symbol, weeks_back)
        if not symbol_prices.empty:
            dates = symbol_prices.index.astype(str)
            st.text_input('No. of days back-test:', value=0, key="backtest_period")
            n_days_back = int(st.session_state.backtest_period)
            end_date = data_retriever.n_days_before(data_retriever.today(), n_days_back)
            symbol_prices_backtest = symbol_prices[symbol_prices.index <= end_date]
            backtest_dates = symbol_prices_backtest.index.astype(str)

def show_candle_chart():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Candle Chart for {symbol}**")

    # Retrieve the symbol prices if not available or update it if the user changes the symbol
    if 'symbol_prices' not in st.session_state or st.session_state.selected_symbol != symbol:
        st.session_state.symbol_prices = data_retriever.get_current_stock_data(symbol, 52 * 5)  # Assuming 5 years of weekly data
        st.session_state.selected_symbol = symbol

    # Check if there are prices to display
    if not st.session_state.symbol_prices.empty:
        symbol_prices = st.session_state.symbol_prices
        dates = symbol_prices.index.astype(str)

        # Create a candlestick graph
        candle_figure = make_subplots(rows=1, cols=1)
        ui.create_candlestick(candle_figure, dates, symbol_prices, symbol, 'Price')

        # Update layout of the figure
        candle_figure.update_layout(
            xaxis_title='Date',
            yaxis_title="Price per Share",
            template='plotly_dark'
        )

        # Display the chart
        st.plotly_chart(candle_figure, use_container_width=True)
    else:
        st.error("No price data available for the selected symbol.")



def show_historical_data():
    # Get symbols and exchange data as with the candle chart
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])

    # Retrieve stock data for the selected ticker
    if 'symbol_prices_hist' not in st.session_state or st.session_state.selected_symbol_hist != symbol:
        st.session_state.symbol_prices_hist = data_retriever.get_current_stock_data(symbol, 52 * 5)  # 5 years of weekly data
        st.session_state.selected_symbol_hist = symbol

    if not st.session_state.symbol_prices_hist.empty:
        symbol_prices_hist = st.session_state.symbol_prices_hist
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**End-of-Day Historical Stock Data for {symbol}**")
            st.dataframe(symbol_prices_hist, width=1000)

        with col2:
            st.markdown(f"**Historical EPS Surprises for {symbol}**")
            earnings_data = get_earnings.get_earnings(symbol)
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
                        size=15,  # Fixed size for estimate markers
                        color='MediumPurple',
                        opacity=0.5,
                        line=dict(
                            color='MediumPurple',
                            width=1
                        )
                    )
                ))

                # Customize the layout
                fig.update_layout(
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

def show_analyst_recommendations():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Stock Analyst Recommendations for {symbol}**")
    if symbol:
        recommendations = recommend.get_rec(symbol)
        if recommendations:
            st.bar_chart(recommendations)
        else:
            st.error("No recommendations data available for the selected symbol.")



def show_latest_news():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Latest News for {symbol}**")
    news_data = get_news.get_stock_news(symbol)
    if news_data.empty:
        st.error("No news data available for the selected symbol.")
    else:
        news_data.set_index("headline", inplace=True)
        st.table(news_data)

def show_trends():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Trends for {symbol}**")

    period = st.slider('Select period for trend analysis:', min_value=7, max_value=365, value=30, step=7)
    symbol_prices = data_retriever.get_current_stock_data(symbol, period)
    if symbol_prices.empty:
        st.error("No data available for the selected period.")
    else:
        fig = make_subplots(rows=1, cols=1)
        # Assuming create_line() is a function from ui module to plot line charts
        ui.create_line(fig, symbol_prices.index, symbol_prices['Close'], "Close price")
        st.plotly_chart(fig, use_container_width=True)

def show_forecast_metrics():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Forecst and Performance Metrics for {symbol}**")

    days_to_forecast = st.slider('Days to Forecast:', min_value=30, max_value=365, value=90)
    symbol_prices = data_retriever.get_current_stock_data(symbol, 365)  # Fetching last year's data

    if symbol_prices.empty:
        st.error("Insufficient data for forecasting.")
    else:
        df = predict.transform_price(symbol_prices)
        model = predict.train_prophet_model(df)
        forecast = predict.make_forecast(model, days_to_forecast)

        fig1 = plot_plotly(model, forecast)
        fig1.update_layout(title='Forecast and Performance Metrics')
        st.plotly_chart(fig1, use_container_width=True)

    actual = df['y']
    predicted = forecast['yhat'][:len(df)]
    metrics = predict.calculate_performance_metrics(actual, predicted)
    st.subheader('Performance Metrics')
    st.write('The metrics below provide a quantitative measure of the model’s accuracy. They include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), with lower values indicating better performance.')

    metrics_data = {
        "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
        "Value": [metrics['MAE'], metrics['MSE'], metrics['RMSE']]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Metric", inplace = True)
    st.table(metrics_df)



def show_ai_analysis():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])

    with st.spinner("Generating AI Analysis..."):
        # All necessary information
        company_basic_info = status.get_basic(symbol)
        prices = data_retriever.get_current_stock_data(symbol, 365)  # Last year's data
        news_data = get_news.get_stock_news(symbol)
        recommendations_data = recommend.get_rec(symbol)

        # Prepare the data for input
        symbol_prices_str = str(prices['Adj Close'].iloc[-1]) if not prices.empty else "N/A"
        news_summary = " ".join(news_data["summary"].tolist()) if not news_data.empty else "N/A"
        recommendations_str = f"Keys: {list(recommendations_data.keys())}, Values: {list(recommendations_data.values())}" if recommendations_data else "N/A"
        
        # Convert company_basic_info to a formatted string if it's a dictionary
        company_basic_str = "Keys: {}, Values: {}".format(company_basic_info.keys(), company_basic_info.values()) if isinstance(company_basic_info, dict) else company_basic_info

        # Generate AI analysis
        st.markdown(genai.generate_gemini_response(
            input_prompt,
            symbol,
            symbol_prices_str,
            company_basic_str,
            news_summary,
            recommendations_str
        ))
