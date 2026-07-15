import pandas as pd
import mplfinance as mpf
import yaml
from pathlib import Path
import os

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
RAW_DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
LABELED_DATA_DIR = Path(__file__).parent.parent / 'data' / 'labeled'
IMAGES_DIR = Path(__file__).parent.parent / 'data' / 'images'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

window_size = config['labeling']['window_size']

def generate_images():
    windows_file = LABELED_DATA_DIR / 'windows.parquet'
    if not windows_file.exists():
        print("Error: Labeled windows file not found. Run labeling.py first.")
        return
        
    windows_df = pd.read_parquet(windows_file)
    
    # Create class directories
    classes = windows_df['label'].unique()
    for cls in classes:
        (IMAGES_DIR / cls).mkdir(parents=True, exist_ok=True)
        
    # Load all raw data into a dict for fast access
    raw_data = {}
    for file in RAW_DATA_DIR.glob('*.parquet'):
        df = pd.read_parquet(file)
        df.set_index('Date', inplace=True)
        ticker = df['Ticker'].iloc[0]
        raw_data[ticker] = df
        
    print(f"Total windows to process: {len(windows_df)}")
    
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=False)
    
    count = 0
    for _, row in windows_df.iterrows():
        ticker = row['ticker']
        label = row['label']
        start_date = row['start_date']
        end_date = row['end_date']
        
        # Format dates for filename
        date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        save_path = IMAGES_DIR / label / f"{ticker}_{date_str}.png"
        
        if save_path.exists():
            continue
            
        ticker_df = raw_data.get(ticker)
        if ticker_df is None:
            continue
            
        # Slice window
        try:
            window_slice = ticker_df.loc[start_date:end_date]
            if len(window_slice) < 5: # Basic check
                continue
                
            # Render chart silently without axes
            mpf.plot(
                window_slice, 
                type='candle',
                style=s,
                axisoff=True,
                savefig=dict(fname=save_path, dpi=100, bbox_inches='tight', pad_inches=0)
            )
            count += 1
            if count % 100 == 0:
                print(f"Generated {count} images...")
        except Exception as e:
            print(f"Error processing {ticker} at {end_date}: {e}")
            
    print(f"Finished generating {count} new images.")

if __name__ == "__main__":
    generate_images()
