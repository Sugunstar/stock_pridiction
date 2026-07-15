import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle

RAW_DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
LABELED_DATA_DIR = Path(__file__).parent.parent / 'data' / 'labeled'
MODELS_DIR = Path(__file__).parent.parent / 'models'

def extract_features(window_df):
    """
    Extract tabular features for the baseline model based on the rolling window.
    Focus on the last few candles where the pattern is formed.
    """
    features = {}
    
    # Analyze the last candle (the pattern candle)
    last_c = window_df.iloc[-1]
    prev_c = window_df.iloc[-2]
    
    body = last_c['Close'] - last_c['Open']
    prev_body = prev_c['Close'] - prev_c['Open']
    
    upper_wick = last_c['High'] - max(last_c['Open'], last_c['Close'])
    lower_wick = min(last_c['Open'], last_c['Close']) - last_c['Low']
    
    total_range = last_c['High'] - last_c['Low']
    if total_range == 0: total_range = 1e-5
    
    features['body_pct'] = abs(body) / total_range
    features['upper_wick_pct'] = upper_wick / total_range
    features['lower_wick_pct'] = lower_wick / total_range
    
    # Position relative to previous candle
    features['close_vs_prev_close'] = (last_c['Close'] - prev_c['Close']) / prev_c['Close']
    features['body_dir'] = 1 if body > 0 else -1
    features['prev_body_dir'] = 1 if prev_body > 0 else -1
    
    # Simple trend calculation over the window (e.g. 20 days)
    features['trend_20d'] = (last_c['Close'] - window_df.iloc[0]['Close']) / window_df.iloc[0]['Close']
    
    return features

def train_baseline():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    windows_file = LABELED_DATA_DIR / 'windows.parquet'
    if not windows_file.exists():
        print("Run labeling.py first.")
        return
        
    windows_df = pd.read_parquet(windows_file)
    
    raw_data = {}
    for file in RAW_DATA_DIR.glob('*.parquet'):
        df = pd.read_parquet(file)
        df.set_index('Date', inplace=True)
        ticker = df['Ticker'].iloc[0]
        raw_data[ticker] = df
        
    print("Extracting features...")
    X_list = []
    y_list = []
    
    for _, row in windows_df.iterrows():
        ticker = row['ticker']
        label = row['label']
        start_date = row['start_date']
        end_date = row['end_date']
        
        ticker_df = raw_data.get(ticker)
        if ticker_df is not None:
            window_slice = ticker_df.loc[start_date:end_date]
            if len(window_slice) >= 2:
                feats = extract_features(window_slice)
                X_list.append(feats)
                y_list.append(label)
                
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Save encoder
    with open(MODELS_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    # Time-based or sequential split is better, but doing random split here for the baseline simplicity.
    # In production, ensure no data leakage.
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    
    print("Training XGBoost baseline model...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, objective='multi:softprob')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n--- Baseline Model Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model.save_model(MODELS_DIR / 'baseline_xgb.json')
    print("\nModel saved to models/baseline_xgb.json")

if __name__ == "__main__":
    train_baseline()
