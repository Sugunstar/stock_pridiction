import pandas as pd
import numpy as np
import talib
import yaml
from pathlib import Path
import os

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
RAW_DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
LABELED_DATA_DIR = Path(__file__).parent.parent / 'data' / 'labeled'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

window_size = config['labeling']['window_size']

def apply_talib_patterns(df):
    """
    Applies TA-Lib pattern recognition to the OHLC data.
    """
    op = df['Open'].values
    hi = df['High'].values
    lo = df['Low'].values
    cl = df['Close'].values
    
    # TA-Lib returns 100 for bullish, -100 for bearish, 0 for none.
    doji = talib.CDLDOJI(op, hi, lo, cl)
    hammer = talib.CDLHAMMER(op, hi, lo, cl)
    engulfing = talib.CDLENGULFING(op, hi, lo, cl)
    shooting_star = talib.CDLSHOOTINGSTAR(op, hi, lo, cl)
    
    df['doji'] = (doji != 0).astype(int)
    df['hammer'] = (hammer != 0).astype(int)
    df['bullish_engulfing'] = (engulfing == 100).astype(int)
    df['bearish_engulfing'] = (engulfing == -100).astype(int)
    df['shooting_star'] = (shooting_star != 0).astype(int)
    
    return df

def generate_labeled_windows():
    LABELED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    all_windows = []
    
    for file in RAW_DATA_DIR.glob('*.parquet'):
        print(f"Processing {file.name}...")
        df = pd.read_parquet(file)
        df = apply_talib_patterns(df)
        
        # Iterate over df to extract windows
        for i in range(window_size, len(df)):
            row = df.iloc[i]
            
            pattern_detected = False
            label = None
            
            if row['doji']:
                label = 'doji'
                pattern_detected = True
            elif row['hammer']:
                label = 'hammer'
                pattern_detected = True
            elif row['bullish_engulfing']:
                label = 'bullish_engulfing'
                pattern_detected = True
            elif row['bearish_engulfing']:
                label = 'bearish_engulfing'
                pattern_detected = True
            elif row['shooting_star']:
                label = 'shooting_star'
                pattern_detected = True
                
            if pattern_detected:
                window_start = df.iloc[i - window_size]['Date']
                window_end = row['Date']
                ticker = row['Ticker']
                
                all_windows.append({
                    'ticker': ticker,
                    'start_date': window_start,
                    'end_date': window_end,
                    'label': label,
                    'end_index': i # to easily slice the dataframe later
                })
                
    windows_df = pd.DataFrame(all_windows)
    
    # Check class balance
    print("\nClass Balance:")
    print(windows_df['label'].value_counts())
    
    # Save the index of windows
    save_path = LABELED_DATA_DIR / 'windows.parquet'
    windows_df.to_parquet(save_path, index=False)
    print(f"\nSaved labeled windows index to {save_path}")

if __name__ == "__main__":
    generate_labeled_windows()
