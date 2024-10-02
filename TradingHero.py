# Standard Library Imports
from datetime import datetime, timedelta
import time
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
import streamlit.components.v1 as components

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
from modules import footer

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
    if 'insights' not in st.session_state:
        st.session_state['insights'] = []

    # Use session_state to track which section was last opened
    if 'last_opened' not in st.session_state:
        st.session_state['last_opened'] = 'Stock Overall Information'

    with st.sidebar:
        if st.sidebar.button('ETF Holdings'):
            st.session_state['last_opened'] = 'ETF Holdings'
        if st.sidebar.button('Stock Overall Information'):
            st.session_state['last_opened'] = 'Stock Overall Information'
        if st.sidebar.button('Historical Stock and EPS Surprises'):
            st.session_state['last_opened'] = 'End-of-Day Historical Stock and EPS Surprises'
        if st.sidebar.button('Stock Analyst Recommendations'):
            st.session_state['last_opened'] = 'Stock Analyst Recommendations'
        if st.sidebar.button('Latest News'):
            st.session_state['last_opened'] = 'Latest News'
        if st.sidebar.button('Time Series Forecasting'):
            st.session_state['last_opened'] = 'Time Series Forecasting'
        # if st.sidebar.button('Trends Forecasting and TradingHero Analysis'):
        #     st.session_state['last_opened'] = 'Trends'
        if st.sidebar.button('Final Equity Report'):
            st.session_state['last_opened'] = 'Final Equity Report'




    # Check which content to display based on what was clicked in the sidebar
    if st.session_state['last_opened'] == 'Stock Overall Information':
        show_overall_information()
    elif st.session_state['last_opened'] == 'End-of-Day Historical Stock and EPS Surprises':
        show_historical_data()
    elif st.session_state['last_opened'] == 'Stock Analyst Recommendations':
        show_analyst_recommendations()
    elif st.session_state['last_opened'] == 'Latest News':
        show_news()  
    elif st.session_state['last_opened'] == 'Time Series Forecasting':
        show_ts() 
    # elif st.session_state['last_opened'] == 'Trends':
    #     show_trends()
    elif st.session_state['last_opened'] == 'ETF Holdings':
        show_etf()
    elif st.session_state['last_opened'] == 'Final Equity Report':
        final_report()


#------------------ 'ETF Holdings'

def show_etf():
    # List of ETFs with their fund names
    etfs = [
        ("SPY", "SPDR S&P 500 ETF Trust"),
        ("IVV", "iShares Core S&P 500 ETF"),
        ("VOO", "Vanguard S&P 500 ETF"),
        ("VTI", "Vanguard Total Stock Market ETF"),
        ("QQQ", "Invesco QQQ Trust Series I"),
        ("VEA", "Vanguard FTSE Developed Markets ETF"),
        ("VUG", "Vanguard Growth ETF"),
        ("VTV", "Vanguard Value ETF"),
        ("IEFA", "iShares Core MSCI EAFE ETF"),
        ("AGG", "iShares Core U.S. Aggregate Bond ETF"),
        ("BND", "Vanguard Total Bond Market ETF"),
        ("IWF", "iShares Russell 1000 Growth ETF"),
        ("IJH", "iShares Core S&P Mid-Cap ETF"),
        ("IJR", "iShares Core S&P Small-Cap ETF"),
        ("VIG", "Vanguard Dividend Appreciation ETF"),
        ("IEMG", "iShares Core MSCI Emerging Markets ETF"),
        ("VWO", "Vanguard FTSE Emerging Markets ETF"),
        ("VXUS", "Vanguard Total International Stock ETF"),
        ("VGT", "Vanguard Information Technology ETF"),
        ("IWM", "iShares Russell 2000 ETF"),
        ("GLD", "SPDR Gold Shares"),
        ("XLK", "Technology Select Sector SPDR ETF"),
        ("VO", "Vanguard Mid-Cap ETF"),
        ("TLT", "iShares 20+ Year Treasury Bond ETF"),
        ("RSP", "Invesco S&P 500 Equal Weight ETF")
    ]

    # Create a dictionary for easy lookup
    etf_dict = {symbol: name for symbol, name in etfs}

    # Dropdown for sample ETFs with display of both symbol and fund name
    selected_etf = st.selectbox(
        "Select an ETF from the list",
        options=[f"{symbol} - {name}" for symbol, name in etfs]
    )

    # Extract the symbol from the selected option
    selected_symbol = selected_etf.split(" - ")[0]

    # Text input for custom ETF symbol
    custom_etf = st.text_input("Or enter a custom ETF symbol")

    # Determine the symbol to use
    symbol = custom_etf if custom_etf else selected_symbol

    finnhub_widget = f"""
    <!-- Finnhub Widget BEGIN -->
    <div style="border: 1px solid #e0e3eb; height: 600px; width: 100%">
        <iframe width="100%" frameBorder="0"
                height="100%"
                src="https://widget.finnhub.io/widgets/etf-holdings?symbol={symbol}&theme=light">
        </iframe>
    </div>
    <div style="width: 100%; text-align: center; margin-top:10px;">
            <a href="https://finnhub.io/" target="_blank" style="color: #1db954;">"co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog"</a>
    </div>
    <!-- Finnhub Widget END -->
    """

    components.html(finnhub_widget, height=600)

    # HTML code for the TradingView ETF heatmap widget
    heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-etf-heatmap.js" async>
      {{
      "dataSource": "AllUSEtf",
      "blockSize": "aum",
      "blockColor": "change",
      "grouping": "asset_class",
      "locale": "en",
      "symbolUrl": "",
      "colorTheme": "light",
      "hasTopBar": false,
      "isDataSetEnabled": false,
      "isZoomEnabled": true,
      "hasSymbolTooltip": true,
      "isMonoSize": false,
      "width": "100%",
      "height": "100%"
    }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """

    components.html(heatmap, height=500)


#------------------ 'Stock Overall Information'

def show_overall_information():
    tradingview_ticker_tape = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <div class="tradingview-widget-copyright">
        <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
        <span class="blue-text">Track all markets on TradingView</span>
        </a>
    </div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
    {
    "symbols": [
    {
        "proName": "FOREXCOM:SPXUSD",
        "title": "S&P 500 Index"
    },
    {
        "proName": "NASDAQ:QQQ",
        "title": "NASDAQ 100 ETF"
    },
    {
        "proName": "FOREXCOM:NSXUSD",
        "title": "US 100 Cash CFD"
    },
    {
        "proName": "FX_IDC:EURUSD",
        "title": "EUR to USD"
    },
    {
        "proName": "BITSTAMP:BTCUSD",
        "title": "Bitcoin"
    },
    {
        "proName": "BITSTAMP:ETHUSD",
        "title": "Ethereum"
    },
    {
        "proName": "FOREXCOM:DJI",
        "title": "Dow Jones"
    },
    {
        "proName": "AMEX:SPY",
        "title": "S&P 500 ETF"
    },
    {
        "proName": "FOREXCOM:USOIL",
        "title": "Crude Oil"
    },
    {
        "proName": "NASDAQ:AAPL",
        "title": "Apple"
    },
    {
        "proName": "NASDAQ:GOOGL",
        "title": "Alphabet (Google)"
    },
    {
        "proName": "NASDAQ:AMZN",
        "title": "Amazon"
    },
    {
        "proName": "NASDAQ:MSFT",
        "title": "Microsoft"
    },
    {
        "proName": "NYSE:BRK.B",
        "title": "Berkshire Hathaway"
    },
    {
        "proName": "NYSE:JNJ",
        "title": "Johnson & Johnson"
    },
    {
        "proName": "NYSE:V",
        "title": "Visa"
    },
    {
        "proName": "NYSE:WMT",
        "title": "Walmart"
    },
    {
        "proName": "NYSE:JPM",
        "title": "JPMorgan Chase"
    }],
    "showSymbolLogo": true,
    "isTransparent": true,
    "displayMode": "adaptive",
    "colorTheme": "light",
    "locale": "en"
    }
    </script>
    </div>
    <!-- TradingView Widget END -->
    """

    # Render the TradingView ticker tape widget in Streamlit
    components.html(tradingview_ticker_tape, height=50) 

    market_status = status.get_status('US')
    st.write("**Market Status Summary:**")
    if market_status["isOpen"]:
        st.write("🟢 The US market is currently open for trading.")
    else:
        st.write("🔴 The US market is currently closed.")
        if market_status["holiday"]:
            st.write(f"Reason: {market_status['holiday']}")
        else:
            st.write("🌃 It is currently outside of regular trading hours.")

    st.markdown("---")

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
   
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        exchange_names = data_retriever.get_exchange_code_names()
        # Right now only limit to 1 stock(US Exchange)
        if len(exchange_names) == 1:
            exchange_name = exchange_names[0]  
            st.write("**Exchange (Currently only support US market):**")
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

        st.text_input('No. of years look-back:', value=1, key="years_back")
        years_back = int(st.session_state.years_back)
        weeks_back = years_back * 52

    st.markdown("---")

    st.session_state['selected_symbol'] = selected_symbol
    symbol = st.session_state.get('selected_symbol', symbols[0])

    # company_basic = status.get_basic(symbol)
    # if st.checkbox('Show Company Basics'):
    #     basics_data = data_retriever.get_current_basics(symbol, data_retriever.today())
    #     metrics = data_retriever.get_basic_detail(basics_data)
    #     st.dataframe(metrics[['Explanation', 'Value']], width=3000)
    symbol_prices = data_retriever.get_current_stock_data(symbol, weeks_back)
    if not symbol_prices.empty:
        dates = symbol_prices.index.astype(str)

    # TradingView Symbol Profile Widget HTML code
    symbol_profile_widget_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <div class="tradingview-widget-copyright">
        <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
        <span class="blue-text">Track all markets on TradingView</span>
        </a>
    </div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
    {{
    "width": 400,
    "height": 550,
    "isTransparent": true,
    "colorTheme": "light",
    "symbol": "{symbol}",
    "locale": "en"
    }}
    </script>
    </div>
    <!-- TradingView Widget END -->
    """

    # Financials Analysis Widget HTML code
    financials_analysis_widget_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-financials.js" async>
    {{
    "isTransparent": true,
    "largeChartUrl": "",
    "displayMode": "regular",
    "width": "400",
    "height": "550",
    "colorTheme": "light",
    "symbol": "{symbol}",
    "locale": "en"
    }}
    </script>
    </div>
    <!-- TradingView Widget END -->
    """

    symbol_info_widget_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
    {{
    "symbol": "{symbol}",
    "width": 550,
    "locale": "en",
    "colorTheme": "light",
    "isTransparent": true
    }}
    </script>
    </div>
    <!-- TradingView Widget END -->
    """

    col1, col2, col3 = st.columns(3)

    with col1:
        components.html(symbol_info_widget_html, height=500)

    with col2:
        components.html(financials_analysis_widget_html, height=500)

    with col3:
        components.html(symbol_profile_widget_html, height=600)

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

    st.caption("""
    Trading Hero AI Technical Summary updates its data on a daily basis, ensuring that you have access to the latest information. 
               However, please note that there is a delay in the data update. 
               This means that the data reflects all activities and changes up to the last completed trading day.
    """)
    st.write("**Trading Hero AI Technical Summary:**")
    with st.spinner("Trading Hero AI Technical analysis is working to generate."):
        progress_bar = st.progress(0)
        
        # Simulate some initial progress
        time.sleep(1)
        progress_bar.progress(10)
        
        prices = symbol_prices.loc[:, "Adj Close"]
        symbol = st.session_state.get('selected_symbol', symbols[0])
        
        tech_prompt = f"""
        You are an equity research analyst tasked with providing a technical summary for various stocks 
        based on their recent price movements and technical indicators. Your analysis should include an evaluation of the stock's trend, 
        its performance....

        You can add some emoji in this report if you want to make it interactive.
        
        Please generate a technical summary (only in English)

        Ticker Symbol: {symbol}
        Current Price: {prices}
        
        Detailed Technical Summary with clear line break for each part of your analysis, 
        don't include date, 
        don't ask for more data so you can do more analysis on any part
        """
        
        # Generate content based on the prompt
        tech_responses = model.generate_content(
            [tech_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        ) 

        def extract_text_from_generation_response(responses):
            return [resp.text for resp in responses]
        
        # Simulate more progress
        time.sleep(1)
        progress_bar.progress(60)
        
        # Extract and format the response
        text_responses = extract_text_from_generation_response(tech_responses)
        tech_analysis = "".join(text_responses)
        
        # Simulate more progress
        time.sleep(1)
        progress_bar.progress(90)
        
        st.markdown(tech_analysis)
        
        # Final progress update
        time.sleep(1)
        progress_bar.progress(100)
        
        footer.add_footer()

        st.session_state['insights'].append({
            'section': 'Stock Overall Information',
            'price': prices,
            'symbol': symbol,
            'summary': tech_analysis
        })


#------------------ 'Historical Stock and EPS Surprises'

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
                        size=[max(10, abs(s) * 10) for s in surprisePercents],  # Dynamically size markers, minimum size of 10
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
                        size=15,  # Fixed size for all estimate markers
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

    st.write("**Trading Hero AI EPS Summary:**")     
    with st.spinner("Trading Hero AI EPS analysis is working to generate."):
        progress_bar = st.progress(0)
        
        # Simulate some initial progress
        time.sleep(1)
        progress_bar.progress(10)
        
        earnings_data = get_earnings.get_earnings(symbol)
        
        earnings_prompt = f"""
        You are a financial analyst tasked with providing a detailed earnings analysis for the company with the ticker symbol {symbol}. 
        The specific earnings data is as follows:{earnings_data}

        Key areas to focus on:

        Earnings Per Share (EPS): Analyze the EPS performance, including comparisons to previous quarters and the same quarter from the previous year.
        Guidance: Review the company's forward-looking statements or guidance and provide an interpretation of future prospects.
        Overall Scoring: Provide an overall score or rating for the company's current status and future outlook on a scale of 1-10. 

        [Detailed Earnings Analysis with clear line breaks for each part of your analysis(only in English)]
        Don't ask for more requirements to do analysis, focus on the data provided.
        You can add some emoji in this report if you want to make it interactive."""
        
        # Simulate more progress
        time.sleep(1)
        progress_bar.progress(30)
        
        eps_report = vertex.eps_response(earnings_prompt, symbol, earnings_data)
        
        # Simulate more progress
        time.sleep(1)
        progress_bar.progress(60)
        
        earnings_summary = eps_report.replace('\n', '  \n')
        
        # Simulate more progress
        time.sleep(1)
        progress_bar.progress(90)
        
        st.markdown(earnings_summary)
        
        # Final progress update
        time.sleep(1)
        progress_bar.progress(100)
        
        footer.add_footer()

        st.session_state['insights'].append({
            'section': 'EPS Summary',
            'summary': earnings_summary
        })

#------------------ 'Stock Analyst Recommendations'

def show_analyst_recommendations():
    symbols = get_active_symbols()
    symbol = st.session_state.get('selected_symbol', symbols[0])

    st.markdown(f"**Stock Analyst Recommendations for {symbol}**")

    with st.spinner("Fetching and analyzing stock analyst recommendations..."):
        progress_bar = st.progress(0)
        
        # Simulate some initial progress
        time.sleep(1)
        progress_bar.progress(10)

        if symbol:
            recommendations = recommend.get_rec(symbol)
            
            # Simulate progress after fetching recommendations
            time.sleep(1)
            progress_bar.progress(30)

            if recommendations:
                st.bar_chart(recommendations)
                
                # Simulate progress after displaying chart
                time.sleep(1)
                progress_bar.progress(50)
            else:
                st.error("No recommendations data available for the selected symbol.")
                progress_bar.progress(100)
                return
        
        recomprompt = """
        You are provided with the following data for one company's stock analyst recommendations:
        Based on this information, please provide Positive Sentiment, Negative Sentiment and the Overall.
        You can add some emoji in this report if you want to make it interactive.
        """
        
        # Generate analysis based on recommendations
        recai_data = recommend.generate_vertexai_recommendresponse(recomprompt, recommendations)
        
        # Simulate progress after generating analysis
        time.sleep(1)
        progress_bar.progress(70)
        
        analyst_summary = recai_data.replace('\n', '  \n')
        
        # Simulate progress after formatting the summary
        time.sleep(1)
        progress_bar.progress(90)
        
        st.markdown(analyst_summary)
        
        # Final progress update
        time.sleep(1)
        progress_bar.progress(100)

        footer.add_footer()

        st.session_state['insights'].append({
            'section': 'Analyst Recommendations',
            'summary': analyst_summary
        })


#------------------ 'Latest News'

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
        
        today_str = data_retriever.today()
        # Convert today's date string to a datetime.date object
        today_date = datetime.strptime(today_str, '%Y-%m-%d').date()

        # Calculate the date 60 days ago
        sixty_days_ago = today_date - timedelta(days=60)

        # Format dates into 'YYYY-MM-DD' string format for function call
        today_formatted = today_date.strftime('%Y-%m-%d')
        sixty_days_ago_formatted = sixty_days_ago.strftime('%Y-%m-%d')

        # Fetch news data
        news_data = get_news.get_stock_news(symbol, sixty_days_ago_formatted, today_formatted)
        
        # Simulate progress after fetching news data
        time.sleep(1)
        progress_bar.progress(30)
        
        if not news_data.empty:
            news_data.set_index("headline", inplace=True)
            st.table(news_data)
            
            # Simulate progress after displaying news table
            time.sleep(1)
            progress_bar.progress(60)
        else:
            st.error("No news data available for the selected symbol.")
            progress_bar.progress(100)
            return
        
        # Fetch all stock news data for AI analysis
        news_data_80 = get_news.get_all_stock_news(symbol, sixty_days_ago_formatted, today_formatted)
        
        # Simulate progress after fetching all news data
        time.sleep(1)
        progress_bar.progress(70)
        
        newsprompt = f"""
        You have been provided with the full text of summaries for recent news articles about a specific company {symbol}. 
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

        # Generate AI analysis based on the news prompt
        newsai_data = get_news.generate_vertexai_newsresponse(newsprompt, news_data_80)
        
        # Simulate progress after generating AI analysis
        time.sleep(1)
        progress_bar.progress(90)
        
        news_summary = newsai_data.replace('\n', '  \n') # Ensure newlines are treated as line breaks in Markdown
        st.markdown(news_summary)
        
        # Final progress update
        time.sleep(1)
        progress_bar.progress(100)

        footer.add_footer()

        st.session_state['insights'].append({
            'section': 'News Analysis',
            'summary': news_summary
        })



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

    # Extract performance metrics
    MAE = metrics['MAE']
    MSE = metrics['MSE']
    RMSE = metrics['RMSE']

    # Summarize future_price data
    future_price = forecast[['ds', 'yhat']]
    future_price_summary = {
        'Start Date': future_price['ds'].iloc[0],
        'End Date': future_price['ds'].iloc[-1],
        'Start Price': future_price['yhat'].iloc[0],
        'End Price': future_price['yhat'].iloc[-1],
        'Average Price': future_price['yhat'].mean(),
        'Price Trend': 'increasing' if future_price['yhat'].iloc[-1] > future_price['yhat'].iloc[0] else 'decreasing'
    }

    # Create the prompt with actual data
    tsprompt = f"""
    You are provided with the following data for {symbol}'s future stock time series analysis:

    - **Forecast Period:** {future_price_summary['Start Date'].date()} to {future_price_summary['End Date'].date()}
    - **Forecasted Start Price:** ${future_price_summary['Start Price']:.2f}
    - **Forecasted End Price:** ${future_price_summary['End Price']:.2f}
    - **Average Forecasted Price:** ${future_price_summary['Average Price']:.2f}
    - **Price Trend:** {future_price_summary['Price Trend']}

    **Performance Metrics:**

    - **Mean Absolute Error (MAE):** {MAE:.2f}
    - **Mean Squared Error (MSE):** {MSE:.2f}
    - **Root Mean Squared Error (RMSE):** {RMSE:.2f}

    These values are derived from the performance metrics using Meta's Prophet for direct forecasting.

    Based on this information, please provide insights. Focus on the future price outlook, and discuss the overall trend rather than short-term fluctuations.

    Add emojis to make your output more interactive. This is a one-time response; don't ask for any follow-up.
    """

    # Prepare metrics_data_str
    metrics_data_str = "MAE: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}".format(MAE, MSE, RMSE)

    # Generate AI analysis
    with st.spinner("Generating Time-Series AI Analysis..."):
        if st.button("Show Trading Hero Time-Series AI Analysis"):
            progress_bar = st.progress(0)
            tsai_data = predict.generate_vertexai_tsresponse(tsprompt, future_price, metrics_data_str)
            sanitized_tsai_data = tsai_data.replace('\n', '  \n')
            time.sleep(1)
            progress_bar.progress(10)
            st.markdown(sanitized_tsai_data)
            footer.add_footer()
            time.sleep(1)
            progress_bar.progress(100)



#------------------ 'Trends'

# def show_trends():
#     symbols = get_active_symbols()
#     symbol = st.session_state.get('selected_symbol', symbols[0])
#     st.markdown(f"**Trends and Forecast for {symbol}**")

#     period = st.slider(label='Select Period for Trend Analysis (Days)', min_value=7, max_value=140, value=14, step=7)
#     days_to_forecast = st.slider('Days to Forecast:', min_value=30, max_value=365, value=90)
#     years_back = st.number_input('No. of years look-back:', value=1, min_value=1, max_value=10)
#     weeks_back = int(years_back * 52)

#     symbol_prices = data_retriever.get_current_stock_data(symbol, weeks_back)

#     if symbol_prices.empty:
#         st.error("No data available for the selected symbol.")
#         return

#     dates = symbol_prices.index.format()

#     # Create the indicator figure for trend analysis
#     indicator_figure = make_subplots(rows=1, cols=1)
#     ui.create_line(indicator_figure, dates, symbol_prices['Close'], "Close Price", color="red")

#     # Volume Weighted Average Price (VWAP)
#     vwap = trends.vwap(symbol_prices)
#     ui.create_line(indicator_figure, dates, vwap, "Volume Weighted Average Price (VWAP)", "VWAP")

#     # Bollinger Bands
#     bollinger_dates, bollinger_low, bollinger_high = trends.bollinger_bands(dates, symbol_prices)
#     ui.create_fill_area(indicator_figure, bollinger_dates, bollinger_low, bollinger_high, "Bollinger Bands")

#     # Linear Regression Trend
#     trend = trends.linear_regression_trend(symbol_prices['Close'], period=period)
#     ui.create_line(indicator_figure, dates[period:], trend, f"Trend: {period} Day", color='blue')

#     # Trend Deviation Area
#     price_deviation_from_trend = [price - trend_val for price, trend_val in zip(symbol_prices['Close'][period:], trend)]
#     deviation_price_offset = [price + dev for price, dev in zip(symbol_prices['Close'][period:], price_deviation_from_trend)]
#     ui.create_fill_area(indicator_figure, dates[period:], trend, deviation_price_offset, f"Trend Deviation: {period} Day Trend", color="rgba(100,0,100,0.3)")

#     # Simple Moving Average (SMA)
#     deltas_sma = trends.sma(symbol_prices['Close'], period)
#     ui.create_line(indicator_figure, dates[period:], deltas_sma, f"SMA {period} Day", color='green')

#     # Bids and Strategies
#     init_val = 1000.0
#     buy_and_hold_val = bidder.buy_and_hold(init_val, symbol_prices['Close'])
#     st.write(f"Initial Value: {init_val}")
#     st.write(f"BUY AND HOLD: {buy_and_hold_val}")

#     bid_dates, bid_vals = trends.bids(dates, symbol_prices, period)
#     strat_val = bidder.buy_rule(init_val, bid_dates, bid_vals, dates, symbol_prices['Close'])

#     ui.create_markers(indicator_figure, bid_dates, bid_vals, "Bids", "Price")
#     st.write(f"STRATEGY: {strat_val}")

#     indicator_figure.update_layout(
#         title="Indicators",
#         xaxis_title='Date',
#         yaxis_title="Price",
#         template='plotly_dark'
#     )
#     st.plotly_chart(indicator_figure, use_container_width=True)

#     st.markdown(
#         f"Trend deviation: indicates the extent to which the price is deviating from its current trajectory. "
#         f"The trajectory is the linear regression of the price over {period} days.")
#     st.markdown('''
#         <ol>
#             <li>Mark all intercepts between the SMA and the price.</li>
#             <li>Calculate the cumulative sum of trend deviations. The cumulative sum is reset when an intercept occurs.</li>
#             <li>When there is an intercept and the cumulative sum is above a threshold, then a price reversal is imminent.</li>
#         </ol>
#     ''', unsafe_allow_html=True)

#     st.markdown(
#             f"If the trend intersects SMA, then reset. "
#             f"When the trend intersects the price, place a bid. If the previous intercept is lower, then buy.")

#     recommendations = recommend.get_rec(symbol)
#     with st.spinner("Model is working to generate."):
#         progress_bar = st.progress(0)
#         if st.button("Show Trading Hero AI Analysis"):
#             company_basic = data_retriever.get_current_basics(symbol, data_retriever.today())
#             prices = symbol_prices.loc[:,"Adj Close"]
#             today_str = data_retriever.today()
#             # Convert today's date string to a datetime.date object
#             today_date = datetime.strptime(today_str, '%Y-%m-%d').date()

#             # Calculate the date 60 days ago
#             sixty_days_ago = today_date - timedelta(days=60)

#             # Format dates into 'YYYY-MM-DD' string format for function call
#             today_formatted = today_date.strftime('%Y-%m-%d')
#             sixty_days_ago_formatted = sixty_days_ago.strftime('%Y-%m-%d')

#             news_data = get_news.get_stock_news(symbol, sixty_days_ago_formatted, today_formatted)["headline"].to_string()
#             # news_data = get_news.get_stock_news(symbol)
#             analyst_rec = "Keys: {}, Values: {}".format(recommendations.keys(), recommendations.values())
#             ai_data = vertex.generate_vertexai_response(input_prompt,symbol,prices,company_basic,news_data,analyst_rec)
#             progress_bar.progress(50)
#             sanitized_ai_data = ai_data.replace('\n', '  \n') # Ensure newlines are treated as line breaks in Markdown
#             st.markdown(sanitized_ai_data)
#             progress_bar.progress(100)

#             footer.add_footer()



def final_report():
    with st.spinner("Trading Hero Final Report Model is working to generate."):
        progress_bar = st.progress(0)
        
        final_report_prompt = """
        You are an equity research analyst tasked with generating a comprehensive equity report for the stock with the ticker symbol. 
        This report is based on several analyses, including technical summary, earnings analysis, analyst recommendations, and news sentiment. 
        You should evaluate the stock's overall performance and future outlook based on the insights provided from each section. 
        The report should be written in a professional yet engaging tone, with clear headings for each section, 
        and use emojis where appropriate to make the report more interactive.

        Your report should include the following sections:

        1. **Technical Summary**: 
        Analyze the stock's recent price movements, key technical indicators (e.g., RSI, moving averages), and trading volume trends. 
        The goal is to provide an understanding of the stock's trend and potential future movement.
        - Example Technical Summary:
            "Although the stock has pulled back from higher prices, [Ticker] remains susceptible to further declines. 
            A reversal of the existing trend looks unlikely at this time. Despite a weak technical condition, there are positive signs. 
            Momentum, as measured by the 9-day RSI, is bullish. Over the last 50 trading sessions, there has been more volume on down days 
            than on up days, indicating that [Ticker] is under distribution, which is a bearish condition. The stock is currently 
            above a falling 50-day moving average. A move below this average could trigger additional weakness in the stock."

        2. **Earnings Analysis**: 
        Evaluate the company's earnings performance based on the following data:
        - Earnings Per Share (EPS): Analyze the company's EPS over the past quarters, comparing with historical performance and expectations.
        - Forward Guidance: Assess the company's future prospects based on the provided guidance.
        - Overall Scoring: Provide a score (1-10) of the company's financial performance and future outlook.

        3. **Analyst Recommendations**: 
        Summarize the stock's current analyst recommendations (e.g., buy, hold, sell) and provide insight into the overall sentiment.
        - Positive Sentiment: [Insert analysis]
        - Negative Sentiment: [Insert analysis]
        - Overall Sentiment: [Insert analysis]

        4. **News Sentiment and Key Events**: 
        Analyze the recent news articles about the company and provide insights into the key trends, strengths and weaknesses, 
        and opportunities and threats facing the company. Include a scoring for the company's current status and future outlook (1-10).
        - Key Trends: Identify the major themes recurring across the news.
        - Strengths and Weaknesses: Summarize the positive and negative factors.
        - Opportunities and Threats: Evaluate external factors impacting the company's future.
        - Overall Scoring: Provide an overall score and explain your rationale, weighing recent news more heavily than older news.

        5. **Final Conclusion and Recommendations**: 
        Based on all the sections above (technical summary, earnings analysis, analyst recommendations, and news sentiment), 
        provide a final evaluation of the stock. Offer AI recommendation (buy, hold, or sell with percentage of confidence), and discuss any risks or opportunities 
        that investors should be aware of. This section should tie together the insights from each section into a cohesive final judgment.

        Make sure the report is easy to read and engaging, using emojis where appropriate to enhance readability. 
        Keep the overall tone professional, and provide clear line breaks for each part of the analysis.
        Do not mention any Date in the report.
        """

        # Simulate initial progress
        time.sleep(1)
        progress_bar.progress(10)

        # Gather all insights into a single string
        if 'insights' in st.session_state:
            combined_insights = ""
            for insight in st.session_state['insights']:
                combined_insights += f"\n\n### {insight['section']}\n{insight.get('summary', '')}\n"
            
            # Simulate progress after gathering insights
            time.sleep(1)
            progress_bar.progress(30)

            # Generate final report based on the combined insights
            report = vertex.final_report(final_report_prompt, combined_insights)
            
            # Simulate progress after generating the report
            time.sleep(1)
            progress_bar.progress(60)

            sanitized_ai_data = report.replace('\n', '  \n')
            
            st.title("Trading Hero Equity Report")
            st.markdown(sanitized_ai_data)
            
            # Final progress update
            time.sleep(1)
            progress_bar.progress(100)

            footer.add_footer()
        else:
            st.markdown("No insights have been gathered yet. Please run from top to bottom of the side menu.")