import yfinance as yf
import pandas as pd
import yaml
import os
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

tickers = config['tickers']
start_date = config['date_range']['start']
end_date = config['date_range']['end']

def ingest_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            continue
            
        # Clean data (forward fill, then drop remaining Nans)
        df = df.ffill().dropna()
        
        # Ensure MultiIndex columns are flattened if using latest yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Keep only required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_cols]
        
        # Add Ticker column
        df['Ticker'] = ticker
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Save to parquet
        save_path = DATA_DIR / f"{ticker}_{start_date}_to_{end_date}.parquet"
        df.to_parquet(save_path, index=False)
        print(f"Saved {ticker} to {save_path}")

if __name__ == "__main__":
    ingest_data()
