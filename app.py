import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime
import requests
from transformers import pipeline
import plotly.express as px

# Streamlit Page Config
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Real-Time Stock Market Sentiment & Forecasting Dashboard")

# Dropdown for Stock Symbol
company_dict = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Infosys (INFY)": "INFY",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT"
}

selected_company = st.selectbox("🔎 Choose a Company to View Data", list(company_dict.keys()))
ticker_symbol = company_dict[selected_company]
forecast_period = st.selectbox("⏳ Forecast Period (days)", [7, 30, 60])

# Fetch stock data
stock_data = yf.Ticker(ticker_symbol)
df = stock_data.history(period='1y').reset_index()

if df.empty or 'Close' not in df:
    st.error("Failed to load stock data.")
else:
    df = df[['Date', 'Close']]
    st.subheader(f"📉 Historical Stock Price for {ticker_symbol}")
    fig = px.line(df, x='Date', y='Close', title=f"{ticker_symbol} Closing Price (Past 1 Year)")
    st.plotly_chart(fig)

    # NEWS API Integration
    st.subheader("📰 Recent News Headlines")
    news_api_key = "c43cd2a454e54c7f9e3e12a395d273a8"  
    news_url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&sortBy=publishedAt&language=en&apiKey={news_api_key}"

    articles = []
    try:
        news_response = requests.get(news_url)
        if news_response.status_code == 200:
            articles = news_response.json().get("articles", [])[:5]
            if articles:
                for i, article in enumerate(articles, 1):
                    title = article.get("title", "No Title")
                    url = article.get("url", "#")
                    st.markdown(f"**{i}. [{title}]({url})**")
            else:
                st.info("No recent news found.")
        else:
            st.warning(f"News API Error: {news_response.status_code}")
    except Exception as e:
        st.error(f"Error fetching news: {e}")

    # Sentiment Analysis using FinBERT
    st.subheader("🧠 Sentiment Analysis of Latest News")
    try:
        sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        if articles:
            for article in articles:
                title = article.get("title", "")
                if title:
                    result = sentiment_model(title)[0]
                    label = result['label']
                    score = result['score']
                    st.write(f"📝 {title}")
                    st.write(f"**Sentiment**: {label} | **Confidence**: {score:.2f}")
                    st.markdown("---")
    except Exception as e:
        st.error(f"Sentiment error: {e}")

    # Forecasting with Prophet
    st.subheader("🔮 Forecast Stock Price for Next 30 Days")
    try:
        df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(inplace=True)

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"{ticker_symbol} Forecast for {forecast_period} Days")
        st.plotly_chart(fig_forecast)
    except Exception as e:
        st.error(f"❌ Forecasting Error: {e}")
