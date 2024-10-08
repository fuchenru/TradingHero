<h1 align="center">

<a href="https://kelvinsa0528.wixsite.com/tradinghero" target="_blank">Trading Hero</a> : Empowering Traders with AI Market Analysis

</h1>

<!-- ![Result](https://i.postimg.cc/cLMsKfTw/Art.png) -->
<p align="center">
<img width="600" alt="TR logo 1" src="https://i.postimg.cc/8zhnNjvs/tr-logo1.png">
</p>

Trading Hero is an advanced stock market analysis tool designed to streamline trading processes, reduce information overload, and enhance decision-making for semi-professional retail traders.

# **📑 Table of Contents**

1. [Project Motivation](#Project-Motivation)
2. [Overview of Trading Hero](#Overview)
   * [Project Diagram](#Project-Diagram)  
   * [Front-End](#Front-End)  
3. [Project Components](#Project-Components) 
   * [Trading Hero AI Technical Summary](#AI-Technical)
   * [Trading Hero AI EPS Summary](#EPS)
   * [Stock Analyst Recommendations](#Recommendations)
   * [Trading Hero AI News Analysis](#News-Analysis)
   * [Trading Hero Time Series Forecasting](#Time-Series)
   * [Trading Hero Final Report](#Final-Report)
4. [Technical Challenges](#Technical-Challenges)
5. [Market Potential](#Market-Potential)
6. [Future Developments](#Future-Developments)
   * [Additional Features](#Additional-Features)
   * [Next Steps Timeline](#Next-Steps)
7. [Conclusion](#Conclusion)
8. [Tools Utilized](#Tools-Utilized)
9. [References](#References)
10. [Acknowledgements/About Us](#Acknowledgements)

# **🎯 Project Motivation** <a name="Project-Motivation"></a>

<p align="center">
<img width="380" alt="Trading Motivation Image" src="Assests/Trading_Motivation.png.png">
</p>

Trading Hero was inspired by the challenges faced by semi-professional retail traders who struggle with managing multiple streams of market data. These traders often rely on various platforms for technical, fundamental, and news analysis, which leads to information overload and inefficiency.

**How Trading Hero helps:**  
Trading Hero integrates all necessary data sources into a unified platform, helping traders make well-informed decisions quickly and accurately.

# **🧠 Overview of Trading Hero** <a name="Overview"></a>

### Project Diagram (TBD) <a name="Project-Diagram"></a>
<img width="785" alt="Project Diagram" src="path-to-your-diagram-image">

### Front-End (TBD) <a name="Front-End"></a>
<img width="785" alt="Front-End" src="path-to-your-frontend-image">

# 🧩 Project Components <a name="Project-Components"></a>

### 📈 Stock Overview Dashboard <a name="AI-Technical"></a>
Users can explore real-time market data, top stock gainers and losers, and perform both fundamental and technical analysis on U.S. stocks. The dashboard features a search functionality that allows users to look up specific stock tickers across NYSE and Nasdaq exchanges, while also offering customizable historical analysis. 

The dashboard integrates financial data and visualizations from TradingView and AI-driven insights to help users make informed decisions. Additionally, users can view detailed financial metrics, company profiles, and generate an AI-powered technical summary for deeper analysis.

<p align="center">
<img width="580" alt="Stock Overview Dashboard" src="Assests/Trading Hero AI Technical Summary.png">
</p>

### 📋 Historical Stock and EPS Surprises  <a name="EPS"></a>
This page provides users with a detailed historical analysis of selected stocks, featuring end-of-day stock data such as open, high, low, close prices, and trading volume. Users can observe key stock performance trends and make data-driven decisions based on past market behavior. Additionally, the Historical EPS Surprises section visualizes a stock’s earnings per share (EPS) performance, highlighting whether the company met, exceeded, or missed analyst expectations. A dedicated Trading Hero AI EPS Analysis button allows users to generate AI-powered insights on the stock’s EPS trends for further analysis.

<p align="center">
<img width="580" alt="Historical Stock and EPS Surprises" src="Assests/Trading Hero AI EPS Summary.png">
</p>

### 💡 Stock Analyst Recommendations <a name="Recommendations"></a> 
This page presents users with a comprehensive overview of analyst recommendations for selected stocks. The page also includes a sentiment analysis that highlights the overall sentiment toward the stock based on the collective recommendations.

<p align="center">
<img width="580" alt="Historical Stock and EPS Surprises" src="Assests/Stock Analyst Recommendations.png">
</p>

### 🔮 Trading Hero AI News Analysis <a name="News-Analysis"></a>
Trading Hero leverages cutting-edge Natural Language Processing (NLP) to perform comprehensive sentiment analysis on a vast number of news articles across different financial domains. This tool is designed to assist investors in making informed decisions by offering insights into the sentiment of articles related to specific stocks.

Overview
The Trading Hero Financial Sentiment Analysis model is trained on a wide variety of financial texts, including corporate reports, earnings call transcripts, and analyst reports. By fine-tuning a pre-trained BERT-based model, the system provides accurate sentiment analysis that helps investors gauge market sentiment towards specific stocks and sectors.

You can explore our model on our Hugging Face page 🤗:
[Trading Hero LLM on Hugging Face](https://huggingface.co/fuchenru/Trading-Hero-LLM )

Model Training and Configuration
To build the sentiment analysis model, we used sequential fine-tuning on eight different datasets. Each dataset represents a unique financial domain or time period, allowing the model to generalize effectively across different types of financial news.

Model Details

Model: FinBERT (trained on a total of 4.9 billion tokens, broken down as follows):
Corporate Reports (10-K & 10-Q): 2.5 billion tokens

Earnings Call Transcripts: 1.3 billion tokens

Analyst Reports: 1.1 billion tokens

Optimizer: AdamW with weight decay

Learning Rate: 2e-5

Model Architecture

Attention Dropout Probability: 0.1 (Applies dropout to attention weights to prevent overfitting during training)

Hidden Layers:

Hidden Size: 768 (dimensionality of hidden layers, standard for BERT models)

Number of Hidden Layers: 12 (for deep learning of complex representations)

Intermediate Size: 3072 (size of the feedforward layers in each transformer block)

Attention Heads: 12 (self-attention mechanism with 12 attention heads to focus on different parts of the input)

Datasets

The model was fine-tuned using a custom dataset of financial communication texts, including corporate reports and news articles. The dataset was split into training, validation, and test sets:

Training Set: 10,918,272 tokens
Validation Set: 1,213,184 tokens
Test Set: 1,347,968 tokens

Evaluation Metrics

The model was evaluated on several metrics to ensure high performance in analyzing financial news sentiment:

Test Accuracy: 90.85%
Test Precision: 92.78%
Test Recall: 90.85%
Test F1 Score: 91.33%

<p align="center">
<img width="680" alt="News-Analysis" src="Assests/sentinment_metrix.png">
</p>

### 🔮 Trading Hero Time Series Forecasting <a name="Time-Series"></a>
Use the Prophet model to predict future stock prices, complete with performance metrics for evaluating forecast accuracy.

### 🔮 Trading Hero Final Report <a name="Final-Report"></a>
Use the Prophet model to predict future stock prices, complete with performance metrics for evaluating forecast accuracy.


# **⚙️ Technical Challenges** <a name="Technical-Challenges"></a>
1. **Data Integration**: Managing real-time updates across multiple data sources.
2. **Scalability**: Ensuring the platform can handle large volumes of data as the user base grows.
3. **AI Model Accuracy**: Continuously improving AI prediction models to minimize errors in forecasts and analysis.

# **🚀 Market Potential** <a name="Market-Potential"></a>
Trading Hero addresses a gap in the market for semi-professional retail traders, offering real-time insights and tools for informed decision-making without the complexity of professional platforms. The application is well-positioned to expand into new markets as demand for AI-powered trading tools grows.

# **🛠️ Future Developments** <a name="Future-Developments"></a>

### Possibilities <a name="Additional-Features"></a>
1. **Extended AI Capabilities**: Enhancing the AI model to provide more detailed financial analysis and trend predictions.
2. **Mobile Application**: Developing a mobile version of Trading Hero for on-the-go access.
3. **Collaborations**: Partnering with financial institutions to provide professional-level data.

### Next Steps Timeline <a name="Next-Steps"></a>
<img width="650" alt="Next Steps" src="path-to-your-timeline-image">

# **🔚 Conclusion** <a name="Conclusion"></a>

Trading Hero empowers traders by integrating essential market data and AI-powered insights into a single platform. This tool is designed to simplify trading, reduce information overload, and enhance decision-making efficiency.

# **🛠️ Tools Utilized** <a name="Tools-Utilized"></a>

|  | Category | Tool(s) |
|----------|----------|----------|
| 1 | Data Visualization | ![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black) |
| 2 | AI Models | ![Prophet](https://img.shields.io/badge/Prophet-%2300C4CC.svg?style=for-the-badge) |
| 3 | Backend | ![Python](https://img.shields.io/badge/python-%23ffffff.svg?style=for-the-badge&logo=python&logoColor=black) |
| 4 | Frontend | ![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |

# 📚 References <a name="References"></a>

# 👤 Acknowledgements/About Us <a name="Acknowledgements"></a>
- Kelvin Hsueh ([@chsuehkelvin](https://github.com/chsuehkelvin))
- Peter Fu Chen  ([@fuchenru](https://github.com/fuchenru))  
- Yaoning Yu  ([@yyu6](https://github.com/yyu6))
- Nathan Chen ([@nathanchen07](https://github.com/nathanchen07))

<p align="center">
<img width="600" alt="TR logo 2" src="https://i.imgur.com/Lw9T6s9.png">
</p>
<!-- ![Result](https://i.imgur.com/Lw9T6s9.png) -->
