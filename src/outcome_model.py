import pandas as pd
import numpy as np
import talib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
import pickle
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
RAW_DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
LABELED_DATA_DIR = Path(__file__).parent.parent / 'data' / 'labeled'
MODELS_DIR = Path(__file__).parent.parent / 'models'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

horizons = config['model']['outcome_horizons']

def build_context_features(df):
    """
    Build context features to augment the pattern label for outcome prediction.
    """
    close = df['Close'].values
    
    # RSI
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_hist'] = macdhist
    
    # Price relative to SMA
    sma_50 = talib.SMA(close, timeperiod=50)
    df['price_vs_sma50'] = (df['Close'] - sma_50) / sma_50
    
    return df

def train_outcome_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    windows_file = LABELED_DATA_DIR / 'windows.parquet'
    if not windows_file.exists():
        print("Run labeling.py first.")
        return
        
    windows_df = pd.read_parquet(windows_file)
    
    print("Building dataset for outcome modeling...")
    
    # We will build one model per horizon, or a single model predicting multiple horizons.
    # For simplicity, let's train separate Logistic Regression models per horizon.
    
    dataset = []
    
    for file in RAW_DATA_DIR.glob('*.parquet'):
        df = pd.read_parquet(file)
        ticker = df['Ticker'].iloc[0]
        
        # Build features
        df = build_context_features(df)
        
        # Filter windows for this ticker
        t_windows = windows_df[windows_df['ticker'] == ticker]
        
        for _, row in t_windows.iterrows():
            end_idx = row['end_index']
            
            # Ensure we have enough future data for horizons
            max_horizon = max(horizons)
            if end_idx + max_horizon >= len(df):
                continue
                
            # Extract features at the pattern completion
            feats = {
                'pattern': row['label'],
                'rsi_14': df.iloc[end_idx]['rsi_14'],
                'macd_hist': df.iloc[end_idx]['macd_hist'],
                'price_vs_sma50': df.iloc[end_idx]['price_vs_sma50']
            }
            
            # Extract targets (1 if price goes up, 0 if down)
            pattern_close = df.iloc[end_idx]['Close']
            
            for h in horizons:
                future_close = df.iloc[end_idx + h]['Close']
                feats[f'target_{h}d'] = 1 if future_close > pattern_close else 0
                
            dataset.append(feats)
            
    dataset_df = pd.DataFrame(dataset).dropna()
    
    if len(dataset_df) == 0:
        print("Not enough data to train outcome model.")
        return
        
    # One-hot encode pattern
    dataset_df = pd.get_dummies(dataset_df, columns=['pattern'], drop_first=False)
    
    feature_cols = [c for c in dataset_df.columns if not c.startswith('target_')]
    
    for h in horizons:
        target_col = f'target_{h}d'
        X = dataset_df[feature_cols]
        y = dataset_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        
        # Calibrate probabilities using Isotonic Regression
        calibrated_lr = CalibratedClassifierCV(estimator=lr, method='isotonic', cv=5)
        calibrated_lr.fit(X_train, y_train)
        
        # Eval
        y_prob = calibrated_lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        print(f"\n--- Outcome Model {h}d Horizon ---")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Brier Score (Calibration): {brier:.4f}")
        
        # Save model
        with open(MODELS_DIR / f'outcome_model_{h}d.pkl', 'wb') as f:
            pickle.dump(calibrated_lr, f)
            
    # Save feature columns to ensure consistency during inference
    with open(MODELS_DIR / 'outcome_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
        
    print("\nOutcome models saved.")

if __name__ == "__main__":
    train_outcome_model()
