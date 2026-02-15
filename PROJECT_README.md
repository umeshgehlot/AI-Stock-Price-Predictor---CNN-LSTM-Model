# 📈 AI Stock Price Predictor - CNN-LSTM Model

A comprehensive web-based stock market prediction application using deep learning (CNN-LSTM) to forecast stock prices and index levels for the next 3 trading days.

## 🌟 Features

### Multi-Asset Support
- **📊 Market Indices**
  - Indian Indices: NIFTY 50, NIFTY Bank, SENSEX, NIFTY IT, NIFTY Auto, NIFTY Pharma, NIFTY FMCG, NIFTY Metal
  - Global Indices: S&P 500, Dow Jones, NASDAQ, Russell 2000, FTSE 100, DAX, Nikkei 225, Hang Seng

- **📈 Individual Stocks**
  - NSE India: All NSE-listed stocks (e.g., RELIANCE, TCS, INFY, HDFCBANK)
  - Global Markets: All major stocks via Yahoo Finance (e.g., AAPL, TSLA, GOOGL, MSFT)

### Key Capabilities
- ✅ Real-time market data from Yahoo Finance
- ✅ AI-powered predictions for next 3 trading days
- ✅ Interactive candlestick charts with prediction overlay
- ✅ Historical price analysis (10-day statistics)
- ✅ Percentage change indicators
- ✅ Modern, responsive web interface

## 🏗️ Architecture

### Model Details
- **Type**: CNN-LSTM Neural Network
- **Input Features**: 
  - Open, High, Low, Close prices
  - Volume
  - Day of the week
- **Training Window**: 10 days
- **Prediction Horizon**: 3 days ahead
- **Training Data**: 10 years of historical data from 52 symbols

### Tech Stack
- **Backend**: TensorFlow 2.16.2, Keras
- **Frontend**: Streamlit (Python)
- **Data Source**: Yahoo Finance API
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 🚀 Installation

### Prerequisites
- Python 3.12
- Windows OS (with Long Path support recommended)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/alexkalinins/cnn-lstm-stock.git
cd cnn-lstm-stock
```

2. **Create virtual environment**
```bash
python -m venv C:\ml_env
```

3. **Install dependencies**
```bash
C:\ml_env\Scripts\pip.exe install tensorflow==2.16.2 tf-keras streamlit plotly yfinance pandas numpy
```

## 💻 Usage

### Option 1: Run via Batch File (Easiest)
```bash
run_app.bat
```

### Option 2: Run via Command Line
```bash
C:\ml_env\Scripts\streamlit.exe run app.py
```

### Option 3: Run via Python Script (Command Line Predictions)
```bash
C:\ml_env\Scripts\python.exe predict.py <TICKER>

# Examples:
C:\ml_env\Scripts\python.exe predict.py AAPL
C:\ml_env\Scripts\python.exe predict.py RELIANCE.NS
```

The web interface will automatically open in your browser at `http://localhost:8501`

## 📱 Web Interface Guide

### Using the App

1. **Select Asset Type**
   - Choose between "Indices" or "Individual Stocks"

2. **Choose Market**
   - For Indices: Select Indian or Global Indices
   - For Stocks: Select NSE (India) or Global markets

3. **Select/Enter Symbol**
   - Indices: Choose from dropdown menu
   - Stocks: Enter ticker symbol (e.g., RELIANCE for NSE, AAPL for global)

4. **Click "Predict"**
   - View real-time price data
   - See AI predictions for next 3 days
   - Analyze interactive price charts
   - Review historical statistics

### Example Symbols

**Indian Stocks (NSE)**
```
RELIANCE    - Reliance Industries
TCS         - Tata Consultancy Services
INFY        - Infosys
HDFCBANK    - HDFC Bank
ICICIBANK   - ICICI Bank
SBIN        - State Bank of India
BHARTIARTL  - Bharti Airtel
KOTAKBANK   - Kotak Mahindra Bank
```

**Global Stocks**
```
AAPL        - Apple Inc.
TSLA        - Tesla Inc.
GOOGL       - Alphabet (Google)
MSFT        - Microsoft
AMZN        - Amazon
NVDA        - NVIDIA
META        - Meta (Facebook)
```

## 📊 Model Performance

The model is trained on:
- **52 symbols** (American and Canadian stocks)
- **10 years** of daily historical data
- **5% validation split**
- **z-score standardization** for each sequence

Three separate models predict:
- Day +1: `models/day1/NEXT1-E19.h5`
- Day +2: `models/day2/NEXT2-E18.h5`
- Day +3: `models/day3/NEXT3-E6.h5`

## 📁 Project Structure

```
cnn-lstm-stock/
├── app.py                  # Streamlit web application
├── predict.py              # Command-line prediction script
├── model1.py               # Model architecture and training
├── download-data.py        # Data download and preprocessing
├── run_app.bat             # Windows batch launcher
├── requirements.txt        # Python dependencies
├── models/                 # Pre-trained models
│   ├── day1/
│   ├── day2/
│   └── day3/
├── logs/                   # TensorBoard logs
└── README.md              # Original documentation
```

## ⚠️ Disclaimer

**IMPORTANT**: The predictions made by this algorithm are for **educational and research purposes only**. 

- ❌ NOT financial advice
- ❌ NOT investment recommendations
- ❌ Past performance does not guarantee future results

Always:
- ✅ Consult with licensed financial advisors
- ✅ Do your own research
- ✅ Understand the risks before investing

## 🔧 Troubleshooting

### Common Issues

**1. TensorFlow Installation Error (Windows Long Path)**
```bash
# Enable Windows Long Path support or use virtual env in shorter path
python -m venv C:\ml_env
```

**2. Model Loading Error**
```python
# Use tf_keras instead of tensorflow.keras
import tf_keras as keras
```

**3. NSE Data Fetch Issues**
```
# App automatically falls back to Yahoo Finance with .NS suffix
# No action needed - this is expected behavior
```

**4. Streamlit Not Opening**
```bash
# Manually open browser to:
http://localhost:8501
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Enhanced data preprocessing (beyond z-score)
- [ ] More balanced training data (bullish vs bearish stocks)
- [ ] Experiment with different sequence lengths
- [ ] Try alternative architectures
- [ ] Add plotting and visualization features
- [ ] Implement backtesting functionality

## 📝 License

This project builds upon [alexkalinins/cnn-lstm-stock](https://github.com/alexkalinins/cnn-lstm-stock)

## 🔗 References

- Lu et al. (2020) - CNN-LSTM Stock Prediction Paper
- Yahoo Finance API Documentation
- TensorFlow/Keras Documentation
- Streamlit Documentation

## 📧 Support

For issues and questions:
- Check existing documentation
- Review troubleshooting section
- Test with known working symbols (AAPL, RELIANCE)

---

**Built with ❤️ using TensorFlow, Streamlit, and Python**

Last Updated: October 4, 2025
