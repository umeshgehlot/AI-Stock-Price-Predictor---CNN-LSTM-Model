# Abstract

This project delivers a three-day stock price forecasting system that couples convolutional and recurrent neural networks within a streamlined data science workflow. Real-time market data is retrieved from Yahoo Finance, processed into supervised learning sequences, and scored by pre-trained CNN-LSTM models. A Streamlit dashboard presents predictions, candlestick visualizations, and contextual analytics to help users explore expected price trajectories across indices and equities. The emphasis is on an end-to-end, reproducible pipeline that bridges data acquisition, deep learning inference, and interactive reporting.

# Introduction

Short-term stock movement prediction is a challenging problem owing to market volatility, noise, and regime shifts. Recent research has shown that hybrid architectures combining convolutional feature extraction with sequence-aware recurrent layers can capture localized price patterns while modeling temporal dependencies. This documentation summarises the design and implementation of the CNN-LSTM Stock Price Predictor, which targets Indian and global indices as well as NSE and international equities. The primary objective is to offer an accessible, configurable application that enables rapid experimentation with deep learning-based forecasts without requiring users to write orchestration code from scratch.

# Methodology

## Data Acquisition
Historical price data (Open, High, Low, Close, Volume) and daily trading metadata are sourced programmatically from Yahoo Finance via `yfinance`. For NSE symbols, the loader automatically appends the `.NS` suffix when needed, ensuring consistent coverage for domestic tickers.

## Preprocessing
- Sliding 10-day windows transform the raw time series into supervised sequences matching the model’s receptive field.
- Features are standardized using z-score scaling on a per-window basis to stabilize training and inference.
- Calendar-derived features (e.g., day of week) are appended to inject seasonality awareness.

## Model Architecture
- Convolutional blocks extract short-term spatial patterns from the multivariate sequences.
- LSTM layers model temporal dependencies and retain multi-step context.
- Three dedicated heads (`models/day1`, `models/day2`, `models/day3`) specialize in predicting the next trading day, day two, and day three horizons respectively.

## Training Strategy
- The baseline model is trained on 10 years of daily data covering 52 diversified symbols to encourage generalization.
- A 5% validation split monitors overfitting, with checkpoints stored as `.h5` artefacts for each forecast horizon.
- TensorBoard logs (under `logs/`) provide visibility into loss curves and convergence dynamics.

## Deployment Workflow
- `download-data.py` refreshes the local dataset when retraining or updating the models.
- `model1.py` encapsulates the CNN-LSTM architecture definition and training routine.
- `predict.py` performs command-line inference, supporting batch predictions for arbitrary tickers.
- `app.py` serves the Streamlit interface, loading the saved weights and rendering analytics for end users.

# Hardware Requirements

- Developer/workstation with a modern 64-bit CPU, 16 GB RAM recommended for retraining workflows.
- Optional NVIDIA GPU (CUDA-capable) to accelerate model retraining; not required for inference.
- Minimum 5 GB free disk space to store historical datasets, trained weights, and logs.

# Software Requirements

- Windows 10/11 with long-path support enabled for TensorFlow installations.
- Python 3.12 (per `requirements.txt`).
- Core Python dependencies: TensorFlow 2.16.2, `tf-keras`, Streamlit, Plotly, `yfinance`, pandas, NumPy.
- Recommended tooling: virtual environment (`venv`) for dependency isolation and Git for version control.

# Source Code

- `app.py`: Streamlit web application that orchestrates data retrieval, prediction, and visualization.
- `predict.py`: CLI utility that loads trained weights and outputs three-day forecasts for a specified ticker.
- `model1.py`: Defines the CNN-LSTM architecture, training loops, and model serialization.
- `download-data.py`: Fetches and preprocesses historical data from Yahoo Finance for retraining or evaluation.
- `models/`: Stores pretrained weights segmented into `day1/`, `day2/`, and `day3/` subdirectories.
- `logs/`: TensorBoard summaries capturing training metrics and model diagnostics.
- `run_app.bat`: Convenience launcher that activates the environment and starts Streamlit on Windows.

# Results

The Streamlit dashboard overlays predicted closing prices on interactive candlestick charts, enabling a side-by-side comparison of historical movements and model outlook. Qualitative evaluation on representative tickers such as `AAPL`, `TSLA`, and `RELIANCE.NS` shows that the CNN-LSTM stack captures short-term directional trends and provides sensible confidence bounds for the next three sessions. Formal quantitative benchmarking (e.g., RMSE, MAPE) is planned as a future enhancement, alongside automated backtesting to assess statistical significance of prediction accuracy.

# References

- Lu, W. et al. (2020). "A CNN-LSTM Model for Stock Price Prediction." Proceedings of the International Conference on Artificial Intelligence.
- Yahoo Finance API Documentation.
- TensorFlow 2.16.2 and Keras Developer Guides.
- Streamlit Documentation.
- Alex Kalinin, "cnn-lstm-stock" GitHub Repository (base project inspiration).
