"""
Model Accuracy Evaluation Script
Evaluates CNN-LSTM stock prediction models using backtesting and accuracy metrics.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try different keras imports for compatibility
try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        import keras

# Model paths
MODEL_1 = './models/day1/NEXT1-E19.h5'
MODEL_2 = './models/day2/NEXT2-E18.h5'
MODEL_3 = './models/day3/NEXT3-E6.h5'

def load_models():
    """Load all three prediction models"""
    print("Loading models...")
    model1 = keras.models.load_model(MODEL_1)
    model2 = keras.models.load_model(MODEL_2)
    model3 = keras.models.load_model(MODEL_3)
    print("Models loaded successfully!")
    return model1, model2, model3

def fetch_data(ticker, period='6mo'):
    """Fetch historical data for evaluation"""
    print(f"Fetching data for {ticker}...")
    data = yf.Ticker(ticker).history(period=period, interval='1d')
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    data['DayOfWeek'] = data.index.dayofweek
    return data

def preprocess_window(window_data):
    """Preprocess a 10-day window for model input"""
    if len(window_data) < 10:
        return None, None, None
    
    cols_to_drop = ['Dividends', 'Stock Splits']
    window = window_data.drop(columns=[col for col in cols_to_drop if col in window_data.columns]).copy()
    
    if window.isnull().values.any():
        return None, None, None
    
    mean = window['Close'].mean()
    std = window['Close'].std()
    
    if std == 0:
        return None, None, None
    
    # Standardize
    for column in window.columns:
        s = window[column].std()
        window[column] = (window[column] - window[column].mean()) / s if s != 0 else 0
    
    model_in = window.values.reshape(1, 10, 1, 6)
    return model_in, mean, std

def predict(models, model_in, mean, std):
    """Make predictions using all three models"""
    model1, model2, model3 = models
    
    out1 = model1.predict(model_in, batch_size=1, verbose=0)[0][0]
    out2 = model2.predict(model_in, batch_size=1, verbose=0)[0][0]
    out3 = model3.predict(model_in, batch_size=1, verbose=0)[0][0]
    
    # Destandardize
    pred1 = out1 * std + mean
    pred2 = out2 * std + mean
    pred3 = out3 * std + mean
    
    return pred1, pred2, pred3

def calculate_metrics(actual, predicted):
    """Calculate accuracy metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {}
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Directional Accuracy (did we predict the right direction?)
    # Compare against previous day's close
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Count': len(actual)
    }

def calculate_directional_accuracy(actual_prices, predicted_prices, base_prices):
    """Calculate directional accuracy - did we predict up/down correctly?"""
    actual_direction = np.array(actual_prices) > np.array(base_prices)
    predicted_direction = np.array(predicted_prices) > np.array(base_prices)
    
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)
    
    return (correct / total) * 100 if total > 0 else 0

def backtest(ticker, models, test_periods=50):
    """Run backtesting on historical data"""
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*60}")
    
    try:
        data = fetch_data(ticker, period='1y')
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    
    results = {
        'day1': {'actual': [], 'predicted': [], 'base': []},
        'day2': {'actual': [], 'predicted': [], 'base': []},
        'day3': {'actual': [], 'predicted': [], 'base': []}
    }
    
    # We need at least 10 days for input + 3 days for validation
    min_required = 13
    total_days = len(data)
    
    if total_days < min_required:
        print(f"Insufficient data: {total_days} days (need at least {min_required})")
        return None
    
    # Calculate how many test points we can create
    max_tests = total_days - min_required + 1
    test_periods = min(test_periods, max_tests)
    
    print(f"Running {test_periods} backtests...")
    
    for i in range(test_periods):
        start_idx = i
        end_idx = start_idx + 10
        
        # Get the 10-day window
        window = data.iloc[start_idx:end_idx]
        
        # Preprocess
        model_in, mean, std = preprocess_window(window)
        
        if model_in is None:
            continue
        
        # Make predictions
        pred1, pred2, pred3 = predict(models, model_in, mean, std)
        
        # Get actual future prices (if they exist)
        base_price = data.iloc[end_idx - 1]['Close']  # Last day of window
        
        if end_idx < len(data):
            actual1 = data.iloc[end_idx]['Close']
            results['day1']['actual'].append(actual1)
            results['day1']['predicted'].append(pred1)
            results['day1']['base'].append(base_price)
        
        if end_idx + 1 < len(data):
            actual2 = data.iloc[end_idx + 1]['Close']
            results['day2']['actual'].append(actual2)
            results['day2']['predicted'].append(pred2)
            results['day2']['base'].append(base_price)
        
        if end_idx + 2 < len(data):
            actual3 = data.iloc[end_idx + 2]['Close']
            results['day3']['actual'].append(actual3)
            results['day3']['predicted'].append(pred3)
            results['day3']['base'].append(base_price)
    
    # Calculate metrics for each prediction horizon
    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    
    all_metrics = {}
    
    for day, day_results in results.items():
        if len(day_results['actual']) > 0:
            metrics = calculate_metrics(day_results['actual'], day_results['predicted'])
            dir_acc = calculate_directional_accuracy(
                day_results['actual'], 
                day_results['predicted'],
                day_results['base']
            )
            metrics['Directional_Accuracy'] = dir_acc
            all_metrics[day] = metrics
            
            print(f"\n{day.upper()} Predictions:")
            print(f"  Samples: {metrics['Count']}")
            print(f"  MAE:  ${metrics['MAE']:.2f}")
            print(f"  RMSE: ${metrics['RMSE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  Directional Accuracy: {dir_acc:.1f}%")
    
    return all_metrics

def main():
    """Main evaluation function"""
    print("="*60)
    print("CNN-LSTM STOCK PREDICTION MODEL ACCURACY EVALUATION")
    print("="*60)
    
    # Load models
    models = load_models()
    
    # Test tickers - mix of indices and stocks
    test_tickers = [
        ("^GSPC", "S&P 500"),
        ("^NSEI", "NIFTY 50"),
        ("AAPL", "Apple Inc."),
        ("RELIANCE.NS", "Reliance Industries"),
        ("MSFT", "Microsoft"),
    ]
    
    all_results = {}
    
    for ticker, name in test_tickers:
        try:
            results = backtest(ticker, models, test_periods=60)
            if results:
                all_results[name] = results
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    if all_results:
        # Calculate averages across all tickers
        avg_metrics = {'day1': [], 'day2': [], 'day3': []}
        
        for name, metrics in all_results.items():
            for day in ['day1', 'day2', 'day3']:
                if day in metrics:
                    avg_metrics[day].append(metrics[day])
        
        print("\nAverage Metrics Across All Tested Symbols:")
        print("-" * 50)
        
        for day in ['day1', 'day2', 'day3']:
            if avg_metrics[day]:
                avg_mape = np.mean([m['MAPE'] for m in avg_metrics[day]])
                avg_dir_acc = np.mean([m['Directional_Accuracy'] for m in avg_metrics[day]])
                avg_mae = np.mean([m['MAE'] for m in avg_metrics[day]])
                
                print(f"\n{day.upper()}:")
                print(f"  Avg MAPE: {avg_mape:.2f}%")
                print(f"  Avg Directional Accuracy: {avg_dir_acc:.1f}%")
                print(f"  Avg MAE: ${avg_mae:.2f}")
        
        print("\n" + "="*60)
        print("INTERPRETATION GUIDE")
        print("="*60)
        print("""
• MAPE (Mean Absolute Percentage Error):
  - < 5%: Excellent
  - 5-10%: Good
  - 10-20%: Acceptable
  - > 20%: Needs improvement

• Directional Accuracy:
  - > 60%: Good predictive power
  - 50-60%: Slight edge over random
  - < 50%: Model may need retraining

• MAE (Mean Absolute Error):
  - Lower is better (context-dependent on price range)
        """)

if __name__ == "__main__":
    main()
