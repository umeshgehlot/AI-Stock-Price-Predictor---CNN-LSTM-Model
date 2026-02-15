import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import tf_keras as keras
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor - CNN-LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        max-width: 100%;
    }
    h1 {
        color: #00ff00;
        text-align: center;
        font-size: 3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Model paths
MODEL_1 = './models/day1/NEXT1-E19.h5'
MODEL_2 = './models/day2/NEXT2-E18.h5'
MODEL_3 = './models/day3/NEXT3-E6.h5'

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        model1 = keras.models.load_model(MODEL_1)
        model2 = keras.models.load_model(MODEL_2)
        model3 = keras.models.load_model(MODEL_3)
        return model1, model2, model3
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def get_nse_data(symbol, period='1mo'):
    """Fetch NSE data via Yahoo Finance (more reliable)"""
    try:
        # Add .NS suffix for NSE stocks on Yahoo Finance
        ticker = symbol.upper() + ".NS"
        data = yf.Ticker(ticker).history(period=period, interval='1d')
        
        if data is not None and not data.empty:
            data['DayOfWeek'] = data.index.dayofweek
            return data
        return None
    except Exception as e:
        st.warning(f"Failed to fetch NSE data: {e}")
        return None

def get_yahoo_data(ticker, period='1mo'):
    """Fetch data from Yahoo Finance"""
    try:
        data = yf.Ticker(ticker).history(period=period, interval='1d')
        if not data.empty:
            data['DayOfWeek'] = data.index.dayofweek
            return data
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def preprocess_data(data):
    """Preprocess data for model input"""
    try:
        # Drop unnecessary columns
        cols_to_drop = ['Dividends', 'Stock Splits']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
        
        # Get last 10 days
        last10 = data.iloc[-10:]
        
        # Check for NaN values
        if last10.isnull().values.any():
            st.error("Invalid data! Please try a different ticker.")
            return None, None, None
        
        mean = last10['Close'].mean()
        std = last10['Close'].std()
        
        # Standardize the data
        for column in last10.columns:
            s = last10[column].std()
            last10[column] = (last10[column] - last10[column].mean()) / s if s != 0 else 0
        
        model_in = last10.values.reshape(1, 10, 1, 6)
        return model_in, mean, std
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None, None

def make_predictions(models, model_in, mean, std):
    """Make predictions using loaded models"""
    try:
        model1, model2, model3 = models
        
        with st.spinner('Making predictions...'):
            output1 = model1.predict(model_in, batch_size=1, verbose=0)[0][0]
            output2 = model2.predict(model_in, batch_size=1, verbose=0)[0][0]
            output3 = model3.predict(model_in, batch_size=1, verbose=0)[0][0]
        
        # Destandardize
        pred1 = round(output1 * std + mean, 2)
        pred2 = round(output2 * std + mean, 2)
        pred3 = round(output3 * std + mean, 2)
        
        return pred1, pred2, pred3
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None, None

def plot_price_chart(data, predictions=None, ticker=""):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Historical'
    ))
    
    # Add predictions if available
    if predictions:
        pred1, pred2, pred3 = predictions
        last_date = data.index[-1]
        
        pred_dates = [
            last_date + timedelta(days=1),
            last_date + timedelta(days=2),
            last_date + timedelta(days=3)
        ]
        pred_values = [pred1, pred2, pred3]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode='markers+lines',
            name='Predictions',
            marker=dict(size=12, color='yellow'),
            line=dict(color='yellow', dash='dash')
        ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price Analysis',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Main App
def main():
    st.markdown("<h1>📈 AI Stock Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Powered by CNN-LSTM Deep Learning Model</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        asset_type = st.radio(
            "Select Asset Type",
            ["📊 Indices", "📈 Individual Stocks"],
            help="Choose between market indices or individual stocks"
        )
        
        if asset_type == "📊 Indices":
            market_type = st.selectbox(
                "Select Market",
                ["Indian Indices", "Global Indices"]
            )
            
            if market_type == "Indian Indices":
                index_options = {
                    "NIFTY 50": "^NSEI",
                    "NIFTY Bank": "^NSEBANK",
                    "SENSEX": "^BSESN",
                    "NIFTY IT": "^CNXIT",
                    "NIFTY Auto": "^CNXAUTO",
                    "NIFTY Pharma": "^CNXPHARMA",
                    "NIFTY FMCG": "^CNXFMCG",
                    "NIFTY Metal": "^CNXMETAL"
                }
                selected_index = st.selectbox("Select Index", list(index_options.keys()))
                ticker = index_options[selected_index]
                ticker_input = selected_index
            else:
                index_options = {
                    "S&P 500": "^GSPC",
                    "Dow Jones": "^DJI",
                    "NASDAQ": "^IXIC",
                    "Russell 2000": "^RUT",
                    "FTSE 100": "^FTSE",
                    "DAX": "^GDAXI",
                    "Nikkei 225": "^N225",
                    "Hang Seng": "^HSI"
                }
                selected_index = st.selectbox("Select Index", list(index_options.keys()))
                ticker = index_options[selected_index]
                ticker_input = selected_index
        else:
            market_type = st.radio(
                "Select Market",
                ["NSE (India)", "Global (Yahoo Finance)"],
                help="Choose between Indian NSE stocks or global stocks"
            )
            
            if market_type == "NSE (India)":
                st.info("📝 Enter NSE symbol (e.g., RELIANCE, TCS, INFY, HDFCBANK)")
                ticker_input = st.text_input("Stock Symbol", "RELIANCE").upper()
                ticker = ticker_input + ".NS"  # Add .NS suffix for Yahoo Finance NSE
            else:
                st.info("📝 Enter symbol (e.g., AAPL, TSLA, GOOGL, MSFT)")
                ticker_input = st.text_input("Stock Symbol", "AAPL").upper()
                ticker = ticker_input
        
        predict_button = st.button("🔮 Predict", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This app uses a CNN-LSTM neural network to predict stock prices for the next 3 days based on:
        - Last 10 days of trading data
        - Open, High, Low, Close, Volume
        - Day of the week patterns
        """)
        
        st.warning("⚠️ **Disclaimer**: Predictions are for educational purposes only. Not financial advice!")
    
    # Load models
    models = load_models()
    
    if models[0] is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Main content area
    if predict_button and ticker_input:
        with st.spinner(f'Fetching data for {ticker_input}...'):
            # Fetch data based on asset type
            if asset_type == "📊 Indices":
                data = get_yahoo_data(ticker, period='1mo')
            elif market_type == "NSE (India)":
                data = get_nse_data(ticker_input, period='1mo')
            else:
                data = get_yahoo_data(ticker, period='1mo')
        
        if data is not None and not data.empty:
            # Display current price info
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # Determine display format
            if asset_type == "📊 Indices":
                if "Indian" in market_type:
                    price_format = lambda x: f"{x:,.2f} pts"
                else:
                    price_format = lambda x: f"{x:,.2f} pts"
            elif market_type == "NSE (India)":
                price_format = lambda x: f"₹{x:,.2f}"
            else:
                price_format = lambda x: f"${x:,.2f}"
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Price" if asset_type == "📈 Individual Stocks" else "Current Level",
                    value=price_format(current_price),
                    delta=f"{change:.2f} ({change_pct:.2f}%)"
                )
            
            with col2:
                st.metric("High", price_format(data['High'].iloc[-1]))
            
            with col3:
                st.metric("Low", price_format(data['Low'].iloc[-1]))
            
            with col4:
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            
            st.markdown("---")
            
            # Preprocess and predict
            model_in, mean, std = preprocess_data(data)
            
            if model_in is not None:
                pred1, pred2, pred3 = make_predictions(models, model_in, mean, std)
                
                if pred1 is not None:
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display predictions
                    st.markdown("### 🔮 Price Predictions")
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        change1 = ((pred1 - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Tomorrow (Day +1)</h3>
                            <h2>{price_format(pred1)}</h2>
                            <p style='color: {"#00ff00" if change1 > 0 else "#ff0000"};'>
                                {change1:+.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        change2 = ((pred2 - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Day After (Day +2)</h3>
                            <h2>{price_format(pred2)}</h2>
                            <p style='color: {"#00ff00" if change2 > 0 else "#ff0000"};'>
                                {change2:+.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col3:
                        change3 = ((pred3 - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Day +3</h3>
                            <h2>{price_format(pred3)}</h2>
                            <p style='color: {"#00ff00" if change3 > 0 else "#ff0000"};'>
                                {change3:+.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Plot chart
                    st.markdown("### 📊 Price Chart with Predictions")
                    fig = plot_price_chart(data, (pred1, pred2, pred3), ticker_input)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional statistics
                    st.markdown("### 📈 Historical Statistics (Last 10 Days)")
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    last_10 = data.iloc[-10:]
                    
                    with stat_col1:
                        st.metric("Average Close", price_format(last_10['Close'].mean()))
                    with stat_col2:
                        st.metric("Volatility", f"{last_10['Close'].std():.2f}")
                    with stat_col3:
                        st.metric("Highest", price_format(last_10['High'].max()))
                    with stat_col4:
                        st.metric("Lowest", price_format(last_10['Low'].min()))
        else:
            st.error(f"Unable to fetch data for {ticker_input}. Please check the symbol and try again.")
    
    else:
        # Welcome screen
        st.markdown("""
        ### 👋 Welcome to AI Stock Price Predictor!
        
        **How to use:**
        1. Select asset type (Indices or Individual Stocks) from the sidebar
        2. Choose your market and symbol
        3. Click "Predict" to see AI-powered predictions for the next 3 days
        
        **📊 Indian Indices:**
        - NIFTY 50, NIFTY Bank, SENSEX, NIFTY IT, NIFTY Auto, NIFTY Pharma
        
        **📊 Global Indices:**
        - S&P 500, Dow Jones, NASDAQ, FTSE 100, DAX, Nikkei 225, Hang Seng
        
        **📈 Popular NSE Stocks:**
        - RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, KOTAKBANK
        
        **📈 Popular Global Stocks:**
        - AAPL (Apple), TSLA (Tesla), GOOGL (Google), MSFT (Microsoft), AMZN (Amazon)
        """)
        
        st.info("👈 Use the sidebar to get started!")
        
        # Display sample predictions section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🎯 Features
            - **Real-time data** from live markets
            - **CNN-LSTM AI Model** for accurate predictions
            - **Interactive charts** with historical data
            - **Support for indices** and individual stocks
            - **Multiple markets** (India & Global)
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Model Details
            - **Architecture**: CNN + LSTM layers
            - **Training Data**: 10 years of historical data
            - **Input Features**: OHLCV + Day of Week
            - **Prediction Horizon**: Next 3 trading days
            - **Accuracy**: Optimized for short-term trends
            """)

if __name__ == "__main__":
    main()
