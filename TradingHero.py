# Standard Library Imports
from datetime import datetime, timedelta
import base64
import certifi
import json
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

# Third-Party Library Imports

# Data Manipulation
import numpy as np
import pandas as pd

# Streamlit
import streamlit as st

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

# Prophet
from prophet.plot import plot_plotly, plot_components_plotly, plot

# Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

# Local Application Imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from modules import bidder
from modules import calculator
from modules import data_retriever
from modules import get_earnings
from modules import get_news
from modules import recommend
from modules import status
from modules import trends
from modules import ui
from modules import vertex
from modules import predict
# import torch in requirements


input_prompt = """
You are an equity research analyst with an uncanny ability to decipher the language of price charts, 
your expertise is crucial in navigating the turbulent seas of financial markets. 
I have provided you with information about a specific stock, including its ticker symbol, 
recent prices, company fundamental information, news, and analyst recommendations. 
Your task is to analyze the stock and provide insights on its recent performance and future prospects.

You can add some emoji in this report if you want to make it interactive.

The first few characters you received is the company's ticker symbol. 

Analysis Guidelines:
1. Company Overview: Begin with a brief overview of the company you are analyzing. Understand its market position, 
recent news, financial health, and sector performance to provide context for your analysis.

2. Fundamental Analysis: Conduct a thorough fundamental analysis of the company. Assess its financial statements, 
including income statements, balance sheets, and cash flow statements. Evaluate key financial ratios 
(e.g., P/E ratio, debt-to-equity, ROE) and consider the company's growth prospects, management effectiveness, competitive positioning, 
and market conditions. This step is crucial for understanding the underlying value and potential of the company.

3. Pattern Recognition: Diligently examine the price chart to identify critical candlestick formations, trendlines, 
and a comprehensive set of technical indicators relevant to the timeframe and instrument in question. 
Pay special attention to recent price movements in the year 2024.

4. Technical Analysis: Leverage your in-depth knowledge of technical analysis principles to interpret the identified patterns and indicators. 
Extract nuanced insights into market dynamics, identify key levels of support and resistance, and gauge potential price movements in the near future.

5. Sentiment Prediction: Based on your technical analysis, predict the likely direction of the stock price. 
Determine whether the stock is poised for a bullish upswing or a bearish downturn. 
Assess the likelihood of a breakout versus a consolidation phase, taking into account the analyst recommendations.

6. Confidence Level: Evaluate the robustness and reliability of your prediction. 
Assign a confidence level based on the coherence and convergence of the technical evidence at hand.

Put more weight on the Pattern Recognition and the news.

Finally, provide your recommendations on whether to Buy, Hold, Sell, Strong Buy, or Strong Sell the stock in the future, 
along with the percentage of confidence you have in your prediction.
"""


vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
model = GenerativeModel("gemini-1.5-pro-002")

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

# temp fix for now
def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)
# url = ("https://financialmodelingprep.com/api/v3/stock/list?apikey=M8vsGpmAiqXW6RxWkSn7a71AvdGHywN8")
# data = get_jsonparsed_data(url)
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
        .css-1d391kg {width: 359px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("https://i.postimg.cc/8zhnNjvs/tr-logo1.png", use_column_width=True)
    st.sidebar.title("Menu 🌐")
    st.image("https://i.imgur.com/WQE6iLY.jpeg", width=805)

    # Use session_state to track which section was last opened
    if 'last_opened' not in st.session_state:
        st.session_state['last_opened'] = 'Overall Information'

    with st.sidebar:
        if st.sidebar.button('Overall Information'):
            st.session_state['last_opened'] = 'Overall Information'
        if st.sidebar.button('Historical Stock and EPS Surprises'):
            st.session_state['last_opened'] = 'End-of-Day Historical Stock and EPS Surprises'
        if st.sidebar.button('Stock Analyst Recommendations'):
            st.session_state['last_opened'] = 'Stock Analyst Recommendations'
        if st.sidebar.button('Latest News'):
            st.session_state['last_opened'] = 'Latest News'
        if st.sidebar.button('Time Series Forecasting'):
            st.session_state['last_opened'] = 'Time Series Forecasting'
        if st.sidebar.button('Trends Forecasting and TradingHero Analysis'):
            st.session_state['last_opened'] = 'Trends'




    # Check which content to display based on what was clicked in the sidebar
    if st.session_state['last_opened'] == 'Overall Information':
        show_overall_information()
    elif st.session_state['last_opened'] == 'End-of-Day Historical Stock and EPS Surprises':
        show_historical_data()
    elif st.session_state['last_opened'] == 'Stock Analyst Recommendations':
        show_analyst_recommendations()
    elif st.session_state['last_opened'] == 'Latest News':
        show_news()  
    elif st.session_state['last_opened'] == 'Time Series Forecasting':
        show_ts() 
    elif st.session_state['last_opened'] == 'Trends':
        show_trends()


def show_overall_information():

    def get_jsonparsed_data(url):
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        return json.loads(data)
    
    # Fetch and display top gainers
    gain_url = ("https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey=M8vsGpmAiqXW6RxWkSn7a71AvdGHywN8")
    gainer_data = get_jsonparsed_data(gain_url)
    st.write("**📈 Today's Top Gainers traded US tickers:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(gainer_data[0]['symbol'], gainer_data[0]['price'], str(gainer_data[0]['changesPercentage'])+'%')
    col2.metric(gainer_data[1]['symbol'], gainer_data[1]['price'], str(gainer_data[1]['changesPercentage'])+'%')
    col3.metric(gainer_data[2]['symbol'], gainer_data[2]['price'], str(gainer_data[2]['changesPercentage'])+'%')
    col4.metric(gainer_data[3]['symbol'], gainer_data[3]['price'], str(gainer_data[3]['changesPercentage'])+'%')
    col5.metric(gainer_data[4]['symbol'], gainer_data[4]['price'], str(gainer_data[4]['changesPercentage'])+'%')



    # Fetch and display top losers
    lose_url = "https://financialmodelingprep.com/api/v3/stock_market/losers?apikey=M8vsGpmAiqXW6RxWkSn7a71AvdGHywN8"
    loser_data = get_jsonparsed_data(lose_url)
    st.write("**📉 Today's Top Losers traded US tickers:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(loser_data[0]['symbol'], loser_data[0]['price'], str(loser_data[0]['changesPercentage'])+'%')
    col2.metric(loser_data[1]['symbol'], loser_data[1]['price'], str(loser_data[1]['changesPercentage'])+'%')
    col3.metric(loser_data[2]['symbol'], loser_data[2]['price'], str(loser_data[2]['changesPercentage'])+'%')
    col4.metric(loser_data[3]['symbol'], loser_data[3]['price'], str(loser_data[3]['changesPercentage'])+'%')
    col5.metric(loser_data[4]['symbol'], loser_data[4]['price'], str(loser_data[4]['changesPercentage'])+'%')
     
    col1, col2 = st.columns(2)
    with col1:
        exchange_names = data_retriever.get_exchange_code_names()
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
        exchange = data_retriever.get_exchange_codes()[exchange_index]

        symbols = get_active_symbols()
        selected_symbol = st.selectbox(
            'Select Stock Ticker:',
            symbols,
            index=symbols.index(st.session_state.get('selected_symbol', 'AAPL')),
            key='overall_info_symbol'
        )

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

# def show_candle_chart():
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

    st.write("**Trading Hero AI Technical Summary:**")
    prices = symbol_prices.loc[:,"Adj Close"]
    symbol = st.session_state.get('selected_symbol', symbols[0])
    text1 = f"""You are an equity research analyst tasked with providing a technical summary for various stocks 
        based on their recent price movements and technical indicators. Your analysis should include an evaluation of the stock\'s trend, 
        its performance, and key technical indicators such as momentum (measured by the RSI), 
        volume trends, and the position relative to moving averages.

        You can add some emoji in this report if you want to make it interactive.
        
        Please generate a technical summary (only in English) that follows the structure and tone of the example provided below:

        Example Technical Summary:
        \"Although the stock has pulled back from higher prices, [Ticker] remains susceptible to further declines. 
        A reversal of the existing trend looks unlikely at this time. Despite a weak technical condition, there are positive signs. 
        Momentum, as measured by the 9-day RSI, is bullish. Over the last 50 trading sessions, 
        there has been more volume on down days than on up days, indicating that [Ticker] is under distribution, 
        which is a bearish condition. The stock is currently above a falling 50-day moving average. 
        A move below this average could trigger additional weakness in the stock. 
        [Ticker] could find secondary support at its rising 200-day moving average.\"

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

    responses = model.generate_content(
    [text1],
    generation_config=generation_config,
    safety_settings=safety_settings,
    stream=True,)

    def extract_text_from_generation_response(responses):
        return [resp.text for resp in responses] 

    # Extract and format the response
    text_responses = extract_text_from_generation_response(responses)
    full_summary = "".join(text_responses)  # Join all parts into one string
    st.write(full_summary)  # Display the formatted summary
    def add_footer():
        st.markdown("""
        ---
        © 2024 Trading Hero. All rights reserved.
                    
        **Disclaimer:** Trading Hero AI can make mistakes. It is not intended as financial advice.
        """, unsafe_allow_html=True)

    add_footer()


def show_historical_data():
    # Get symbols and exchange data as with the candle chart
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])

    # Retrieve stock data for the selected ticker
    if 'symbol_prices_hist' not in st.session_state or st.session_state.selected_symbol_hist != symbol:
        st.session_state.symbol_prices_hist = data_retriever.get_current_stock_data(symbol, 52 * 5)  # 5 years of weekly data
        st.session_state.selected_symbol_hist = symbol

    if not st.session_state.symbol_prices_hist.empty:
        symbol_prices_hist = st.session_state.symbol_prices_hist[::-1]
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


def show_analyst_recommendations():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])

    st.markdown(f"**Stock Analyst Recommendations for {symbol}**")
    # st.markdown(f'**Company Rating:{get_jsonparsed_data(url)[0]['rating']} with RatingScore:**')
    
    if symbol:
        recommendations = recommend.get_rec(symbol)
        if recommendations:
            # st.subheader('Stock Analyst Recommendations')
            st.bar_chart(recommendations)
        else:
            st.error("No recommendations data available for the selected symbol.")
    
    company_ratings = get_jsonparsed_data(f"https://financialmodelingprep.com/api/v3/rating/{symbol}?apikey=M8vsGpmAiqXW6RxWkSn7a71AvdGHywN8")[0]
    # if company_ratings:
    #     st.markdown(f"**Company Rating for {symbol}**")
    #     # Parsing the data into a structured format
    # ratings_info = [
    #     {"Category": "Overall", "Recommendation": company_ratings["ratingRecommendation"], "Score": company_ratings["ratingScore"]},
    #     {"Category": "DCF(Discounted Cash Flow)", "Recommendation": company_ratings["ratingDetailsDCFRecommendation"], "Score": company_ratings["ratingDetailsDCFScore"], 'Usage': 'Used to determine the intrinsic value of an investment, considering the time value of money.', 'Description': 'A valuation method used to estimate the value of an investment based on its expected future cash flows. The approach involves forecasting future cash flows and discounting them to their present value using a rate that reflects their risk.'},
    #     {"Category": "ROE(Return on Equity)", "Recommendation": company_ratings["ratingDetailsROERecommendation"], "Score": company_ratings["ratingDetailsROEScore"], 'Usage': "Helps in comparing the profitability of companies within the same industry.", 'Description': "A measure of a company's profitability that reveals how much profit a company generates with the money shareholders have invested. It is calculated as Net Income divided by Shareholders' Equity."},
    #     {"Category": "ROA(Return on Assets)", "Recommendation": company_ratings["ratingDetailsROARecommendation"], "Score": company_ratings["ratingDetailsROAScore"], 'Usage': "Assesses how efficient management is at using assets to generate earnings.", 'Description': "An indicator of how profitable a company is relative to its total assets, calculated by dividing Net Income by Total Assets."},
    #     {"Category": "DE(Debt to Equity)", "Recommendation": company_ratings["ratingDetailsDERecommendation"], "Score": company_ratings["ratingDetailsDEScore"], 'Usage': "Provides insight into a company's leverage and financial health.", "Description": "A ratio indicating the relative proportion of shareholders' equity and debt used to finance a company's assets. Calculated as Total Liabilities divided by Shareholders' Equity."},
    #     {"Category": "PE(Price to Earnings)", "Recommendation": company_ratings["ratingDetailsPERecommendation"], "Score": company_ratings["ratingDetailsPEScore"], 'Usage': "Commonly used by investors to evaluate the market value of a stock compared to the company's earnings.", "Description": "A ratio for valuing a company that measures its current share price relative to its per-share earnings. Calculated as Market Value per Share divided by Earnings per Share (EPS)."},
    #     {"Category": "PB(Price to Book)", "Recommendation": company_ratings["ratingDetailsPBRecommendation"], 'Usage': "Indicates what investors are prepared to pay for each dollar of book value equity.", "Score": company_ratings["ratingDetailsPBScore"], 'Description': "A ratio used to compare a firm's market capitalization to its book value. It is derived by dividing the stock's price per share by book value per share."}
    # ]

    # Creating DataFrame
    # ratings_df = pd.DataFrame(ratings_info)
    # st.dataframe(ratings_df.set_index('Category'), width = 1200)  # Displaying company ratings as JSON
    # AI analysis
    recomprompt = """
        You are provided with the following data for one company's stock analyst recommendations:
        Based on this information, please provide Positive Sentiment, Negative Sentiment and the Overall.
        You can add some emoji in this report if you want to make it interactive.
        """
    recai_data = recommend.generate_vertexai_recommendresponse(recomprompt, recommendations)
    sanitized_recai_data = recai_data.replace('\n', '  \n') # Ensure newlines are treated as line breaks in Markdown
    st.markdown(sanitized_recai_data)
    def add_footer():
        st.markdown("""
        ---
        © 2024 Trading Hero. All rights reserved.
                    
        **Disclaimer:** Trading Hero AI can make mistakes. It is not intended as financial advice.
        """, unsafe_allow_html=True)

    add_footer()

def show_news():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    
    st.markdown(f"**News Analysis for {symbol}**")
    # Information about the NLP model and AI News Analysis
    st.caption("""
    Trading Hero utilizes AI News Analysis to leverage state-of-the-art Natural Language Processing (NLP) model for comprehensive analysis 
    on massive volumes of news articles across diverse domains. This tool helps in making informed investment decisions 
    by providing insights into the sentiment of news articles related to specific stocks.

    **Trading Hero Financial Sentiment Analysis**
    You can find our model on [Hugging Face](https://huggingface.co/fuchenru/Trading-Hero-LLM)
    """)
    with st.spinner("Trading Hero AI News analysis is working to generate."):
        progress_bar = st.progress(0)
        if st.checkbox('Show Latest News'):
            today_str = data_retriever.today()
            # Convert today's date string to a datetime.date object
            today_date = datetime.strptime(today_str, '%Y-%m-%d').date()

            # Calculate the date 60 days ago
            sixty_days_ago = today_date - timedelta(days=60)

            # Format dates into 'YYYY-MM-DD' string format for function call
            today_formatted = today_date.strftime('%Y-%m-%d')
            sixty_days_ago_formatted = sixty_days_ago.strftime('%Y-%m-%d')

            news_data = get_news.get_stock_news(symbol, sixty_days_ago_formatted, today_formatted)
            if not news_data.empty:
                news_data.set_index("headline", inplace=True)
                progress_bar.progress(50)
                st.table(news_data)
                progress_bar.progress(100)
            else:
                st.error("No news data available for the selected symbol.")
                progress_bar.progress(100)
            # AI analysis
            news_data_80 = get_news.get_all_stock_news(symbol, sixty_days_ago_formatted, today_formatted)
            newsprompt = f"""
            You have been provided with the full text of summaries for recent news articles about a specific company{symbol}. 
            Utilize this data to conduct a detailed analysis of the company's current status and future outlook. 
            You can add some emoji in this report if you want to make it interactive.
            Please include the following elements in your analysis:
            
            1. Key Trends: Analyze the recurring topics and themes across the summaries to identify prevalent trends.
            2. Strengths and Weaknesses: Assess the positive and negative attributes of the company as highlighted by the news articles.
            3. Opportunities and Threats: Evaluate external factors that could influence the company positively or negatively in the future.
            4. Overall Scoring: Provide an overall score or rating for the company's current status and future outlook on a scale of 1-10. 
            Please justify and explain your scoring rationale in detail, drawing evidence from the specific details, facts, 
            and narratives portrayed across the news summaries. Your scoring should encompass both the company's present circumstances 
            as well as the projected trajectory factoring in future risks and prospects. Put more weights on most recent news sentiments, and less
            weights on the least recent news.

            The provided summaries contain all necessary details to perform this comprehensive review.  
            Don't include any of your suggestion on if I can provide any more data to you. Make the summary as concise as possible.
            """

            newsai_data = get_news.generate_vertexai_newsresponse(newsprompt, news_data_80)
            sanitized_newsai_data = newsai_data.replace('\n', '  \n') # Ensure newlines are treated as line breaks in Markdown
            st.markdown(sanitized_newsai_data)
            def add_footer():
                st.markdown("""
                ---
                © 2024 Trading Hero. All rights reserved.
                            
                **Disclaimer:** Trading Hero AI can make mistakes. It is not intended as financial advice.
                """, unsafe_allow_html=True)

            add_footer()


def show_ts():
    """Display the time series analysis and AI-generated insights."""
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Trends and Forecast for {symbol}**")

    # User inputs for trend analysis and forecasting
    period = st.slider(label='Select Period for Trend Analysis (Days)', min_value=7, max_value=140, value=14, step=7)
    days_to_forecast = st.slider('Days to Forecast:', min_value=30, max_value=365, value=90)
    years_back = st.number_input('No. of years look-back:', value=1, min_value=1, max_value=10)
    weeks_back = int(years_back * 52)

    # Fetch and transform stock data
    symbol_prices = data_retriever.get_current_stock_data(symbol, weeks_back)
    with st.spinner('Fetching data and training model...'):
        df = predict.transform_price(symbol_prices)
        model = predict.train_prophet_model(df)
        forecast = predict.make_forecast(model, days_to_forecast)

    # Display forecast plot
    st.subheader('Trading Hero Forecasting')
    st.write("""
        The plot below visualizes the forecasted stock prices using Trading Hero's time-series algorithm.
        Our tool handles complexities such as seasonal variations and missing data automatically.
        The forecast includes trend lines and confidence intervals, providing a clear visual representation of expected future values and the uncertainty around these predictions.
    """)

    fig1 = plot_plotly(model, forecast)
    fig1.update_traces(marker=dict(color='red'), line=dict(color='white'))
    st.plotly_chart(fig1)

    # Calculate and display performance metrics
    actual = df['y']
    predicted = forecast['yhat'][:len(df)]
    metrics = predict.calculate_performance_metrics(actual, predicted)
    st.subheader('Performance Metrics')

    metrics_data = {
        "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
        "Value": [metrics['MAE'], metrics['MSE'], metrics['RMSE']]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Metric", inplace=True)
    st.table(metrics_df)

    # AI analysis
    future_price = forecast[['ds', 'yhat']]
    metrics_data_str = "Keys: {}, Values: {}".format(metrics.keys(), metrics.values())
    tsprompt = """
    You are provided with the following data for one company's future stock time series analysis:
    - Future Price (Focus on the overall future trend, not short-term fluctuations)
    - Performance Metrics:
    - MAE: 
    - MSE: 
    - RMSE: 

    These values are derived from the Performance Metrics using Meta's Prophet for direct forecasting.
    Based on this information, please provide insights. Talk more on Future Price overlook.
    Add emoji to make your output more interactive. This is one time response, don't ask for any follow up.
    """
    tsai_data = predict.generate_vertexai_tsresponse(tsprompt, future_price, metrics_data_str)
    with st.spinner("Generating Time-Series AI Analysis..."):
        progress_bar = st.progress(0)
        if st.button("Show Trading Hero Time-Series AI Analysis"):
            progress_bar.progress(50)
            st.markdown(tsai_data)
            progress_bar.progress(100)
            def add_footer():
                st.markdown("""
                ---
                © 2024 Trading Hero. All rights reserved.
                            
                **Disclaimer:** Trading Hero AI can make mistakes. It is not intended as financial advice.
                """, unsafe_allow_html=True)

            add_footer()


def show_trends():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])
    st.markdown(f"**Trends and Forecast for {symbol}**")

    period = st.slider(label='Select Period for Trend Analysis (Days)', min_value=7, max_value=140, value=14, step=7)
    days_to_forecast = st.slider('Days to Forecast:', min_value=30, max_value=365, value=90)
    years_back = st.number_input('No. of years look-back:', value=1, min_value=1, max_value=10)
    weeks_back = int(years_back * 52)

    symbol_prices = data_retriever.get_current_stock_data(symbol, weeks_back)

    if symbol_prices.empty:
        st.error("No data available for the selected symbol.")
        return

    dates = symbol_prices.index.format()

    # Create the indicator figure for trend analysis
    indicator_figure = make_subplots(rows=1, cols=1)
    ui.create_line(indicator_figure, dates, symbol_prices['Close'], "Close Price", color="red")

    # Volume Weighted Average Price (VWAP)
    vwap = trends.vwap(symbol_prices)
    ui.create_line(indicator_figure, dates, vwap, "Volume Weighted Average Price (VWAP)", "VWAP")

    # Bollinger Bands
    bollinger_dates, bollinger_low, bollinger_high = trends.bollinger_bands(dates, symbol_prices)
    ui.create_fill_area(indicator_figure, bollinger_dates, bollinger_low, bollinger_high, "Bollinger Bands")

    # Linear Regression Trend
    trend = trends.linear_regression_trend(symbol_prices['Close'], period=period)
    ui.create_line(indicator_figure, dates[period:], trend, f"Trend: {period} Day", color='blue')

    # Trend Deviation Area
    price_deviation_from_trend = [price - trend_val for price, trend_val in zip(symbol_prices['Close'][period:], trend)]
    deviation_price_offset = [price + dev for price, dev in zip(symbol_prices['Close'][period:], price_deviation_from_trend)]
    ui.create_fill_area(indicator_figure, dates[period:], trend, deviation_price_offset, f"Trend Deviation: {period} Day Trend", color="rgba(100,0,100,0.3)")

    # Simple Moving Average (SMA)
    deltas_sma = trends.sma(symbol_prices['Close'], period)
    ui.create_line(indicator_figure, dates[period:], deltas_sma, f"SMA {period} Day", color='green')

    # Bids and Strategies
    init_val = 1000.0
    buy_and_hold_val = bidder.buy_and_hold(init_val, symbol_prices['Close'])
    st.write(f"Initial Value: {init_val}")
    st.write(f"BUY AND HOLD: {buy_and_hold_val}")

    bid_dates, bid_vals = trends.bids(dates, symbol_prices, period)
    strat_val = bidder.buy_rule(init_val, bid_dates, bid_vals, dates, symbol_prices['Close'])

    ui.create_markers(indicator_figure, bid_dates, bid_vals, "Bids", "Price")
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

    recommendations = recommend.get_rec(symbol)
    with st.spinner("Model is working to generate."):
        progress_bar = st.progress(0)
        if st.button("Show Trading Hero AI Analysis"):
            company_basic = data_retriever.get_current_basics(symbol, data_retriever.today())
            prices = symbol_prices.loc[:,"Adj Close"]
            today_str = data_retriever.today()
            # Convert today's date string to a datetime.date object
            today_date = datetime.strptime(today_str, '%Y-%m-%d').date()

            # Calculate the date 60 days ago
            sixty_days_ago = today_date - timedelta(days=60)

            # Format dates into 'YYYY-MM-DD' string format for function call
            today_formatted = today_date.strftime('%Y-%m-%d')
            sixty_days_ago_formatted = sixty_days_ago.strftime('%Y-%m-%d')

            news_data = get_news.get_stock_news(symbol, sixty_days_ago_formatted, today_formatted)["headline"].to_string()
            # news_data = get_news.get_stock_news(symbol)
            analyst_rec = "Keys: {}, Values: {}".format(recommendations.keys(), recommendations.values())
            ai_data = vertex.generate_vertexai_response(input_prompt,symbol,prices,company_basic,news_data,analyst_rec)
            progress_bar.progress(50)
            sanitized_ai_data = ai_data.replace('\n', '  \n') # Ensure newlines are treated as line breaks in Markdown
            st.markdown(sanitized_ai_data)
            progress_bar.progress(100)
            def add_footer():
                st.markdown("""
                ---
                © 2024 Trading Hero. All rights reserved.
                            
                **Disclaimer:** Trading Hero AI can make mistakes. It is not intended as financial advice.
                """, unsafe_allow_html=True)

            add_footer()
