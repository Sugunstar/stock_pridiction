# CandleSense Developer Guide

Welcome to the CandleSense developer documentation. This guide is intended to help developers understand the system architecture, how each file and function works, and how to tweak or extend the machine learning models.

## System Architecture Overview

CandleSense is built on a decoupled two-model architecture:
1. **Pattern Classifier (CNN)**: Analyzes a candlestick chart image and classifies it into known patterns (Doji, Hammer, Engulfing, etc.).
2. **Outcome Estimator (XGBoost/Logistic Regression)**: Takes the detected pattern along with market context (trend, RSI, MACD, Volume) to estimate the forward-return probability.

This separation ensures that pattern detection (a visual/shape task) is decoupled from market prediction (a statistical/time-series task).

---

## Codebase Structure & File Explanations

### 1. Data Pipeline
* **`src/ingest.py`**
  * **Purpose**: Pulls historical OHLCV data from Yahoo Finance (`yfinance`).
  * **Key Functions**:
    * `fetch_data(ticker, start_date, end_date)`: Fetches data for a specific ticker.
    * `cache_data(df, ticker)`: Saves the dataframe as a Parquet file in `data/raw/`.
  * **Tweaking**: Modify this to pull intraday data (1h, 15m) or integrate other data sources (like Alpaca or Binance for crypto).

* **`src/labeling.py`**
  * **Purpose**: Auto-labels historical data using TA-Lib to avoid manual labeling.
  * **Key Functions**:
    * `apply_talib_patterns(df)`: Runs functions like `CDLDOJI`, `CDLHAMMER`, etc.
    * `extract_windows(df, window_size=20)`: Extracts the `N` preceding candles for each pattern detected to provide visual context.
  * **Tweaking**: 
    * Change `window_size` (default 15-20) to give the CNN more or less context. 
    * Adjust the `penetration` parameter in TA-Lib or add manual rule-based filters if TA-Lib generates noisy labels.

* **`src/chart_gen.py`**
  * **Purpose**: Converts the OHLCV rolling windows into candlestick chart images using `mplfinance`.
  * **Key Functions**:
    * `generate_image(window_df, save_path)`: Renders a clean, noiseless candlestick chart (e.g., 224x224 PNG).
  * **Tweaking**: Alter the image dimensions, change candle colors, or add a volume subplot if you want the CNN to learn volume-price relationships.

### 2. Modeling
* **`src/baseline_model.py`**
  * **Purpose**: A fast tabular baseline (XGBoost/LightGBM) to validate if patterns carry learnable signals before training the CNN.
  * **Key Functions**:
    * `extract_features(window_df)`: Calculates body size, wick ratios, etc.
    * `train_baseline(X, y)`: Trains the XGBoost model and logs accuracy/recall.
  * **Tweaking**: Add more engineered features (e.g., ATR, Bollinger Bands).

* **`src/cnn_model.py`**
  * **Purpose**: The core deep learning model for pattern classification.
  * **Key Functions**:
    * `build_model(num_classes=5)`: Loads a pretrained ResNet18 or EfficientNet-B0 and modifies the classification head.
    * `train_epoch() / eval_epoch()`: Standard PyTorch training loops.
  * **Tweaking**: 
    * **Model Capacity**: Upgrade to ResNet50 or EfficientNet-B3 if the model underfits.
    * **Augmentation**: Add slight brightness/contrast adjustments or vertical scaling. *Never use horizontal flips, as it reverses time!*
    * **Imbalance**: Tweak class weights in the CrossEntropyLoss to focus more on rare patterns like Shooting Stars.

* **`src/outcome_model.py`**
  * **Purpose**: Predicts the probability of price movement given the pattern and context.
  * **Key Functions**:
    * `compute_forward_returns(df, horizons=[1, 3, 5])`: Calculates future returns.
    * `build_context_features(df)`: Calculates RSI, MACD, and price relative to moving averages.
    * `train_outcome_model(X, y)`: Trains a calibrated classifier (e.g., Logistic Regression).
  * **Tweaking**: 
    * Adjust the trading horizons (e.g., 10-day or 20-day returns).
    * Experiment with gradient boosting instead of logistic regression for higher accuracy.

### 3. Validation & Production
* **`src/backtest.py`**
  * **Purpose**: Simulates a trading strategy based on the outcome model's confidence.
  * **Tweaking**: Implement stop-loss, take-profit, or position sizing logic (e.g., Kelly Criterion) to make the backtest more realistic.

* **`src/inference.py`**
  * **Purpose**: The end-to-end pipeline wrapper. Takes a ticker, pulls the latest data, generates the chart, runs the CNN, and queries the outcome model.

* **`api/main.py` & `ui/app.py`**
  * **Purpose**: The FastAPI backend and Streamlit frontend.
  * **Tweaking**: Add endpoints for portfolio/watchlist scanning.

---

## Important Guidelines for Tweaking the Model

1. **Avoid Data Leakage**: When training the CNN or Outcome model, **always split train/test sets by time** (e.g., train on 2018-2023, test on 2024-2025) or by holding out entire tickers. Random splitting will leak future data due to the rolling window approach.
2. **Handle Class Imbalance**: Dojis and Hammers will dominate your dataset. Use weighted loss functions or undersampling to ensure the model learns Engulfing patterns effectively.
3. **Calibrate Probabilities**: The outcome model must use Platt Scaling or Isotonic Regression so that a predicted "60% confidence" actually matches a ~60% empirical win rate.
4. **Statistical Honesty**: Always present the **sample size** alongside predictions in the UI. A 100% win rate over 2 samples is useless compared to a 60% win rate over 500 samples.
