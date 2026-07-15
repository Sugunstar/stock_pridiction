import pandas as pd
import yfinance as yf
import torch
import mplfinance as mpf
from pathlib import Path
import os
import yaml
import pickle
from PIL import Image
from torchvision import transforms
from .cnn_model import build_model
from .outcome_model import build_context_features
import io
import base64

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
MODELS_DIR = Path(__file__).parent.parent / 'models'
DATA_DIR = Path(__file__).parent.parent / 'data'

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    
window_size = config['labeling']['window_size']
horizons = config['model']['outcome_horizons']
image_size = config['model']['image_size']

class InferencePipeline:
    def __init__(self):
        # Load CNN
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Note: In a real scenario you need the class mapping used during training.
        # Assuming alphabetical order of folders: bearish_engulfing, bullish_engulfing, doji, hammer, shooting_star
        self.classes = ['bearish_engulfing', 'bullish_engulfing', 'doji', 'hammer', 'shooting_star']
        
        self.cnn = build_model(len(self.classes)).to(self.device)
        cnn_path = MODELS_DIR / 'cnn_resnet18.pth'
        if cnn_path.exists():
            self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load Outcome Models
        self.outcome_models = {}
        for h in horizons:
            model_path = MODELS_DIR / f'outcome_model_{h}d.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as file:
                    self.outcome_models[h] = pickle.load(file)
                    
        feats_path = MODELS_DIR / 'outcome_features.pkl'
        if feats_path.exists():
            with open(feats_path, 'rb') as file:
                self.feature_cols = pickle.load(file)
        else:
            self.feature_cols = []
            
    def predict(self, ticker):
        # 1. Pull data
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            return {"error": f"No data found for {ticker}"}
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.ffill().dropna()
        if len(df) < window_size + 50: # 50 for SMA50
            return {"error": "Not enough data points."}
            
        # 2. Render chart (latest window)
        window_df = df.iloc[-window_size:]
        
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s  = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=False)
        
        buf = io.BytesIO()
        mpf.plot(
            window_df, 
            type='candle',
            style=s,
            axisoff=True,
            savefig=dict(fname=buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        )
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # 3. CNN prediction
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.cnn(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)
            
        pattern = self.classes[predicted.item()]
        cnn_confidence = conf.item()
        
        # 4. Outcome model
        df = build_context_features(df)
        last_row = df.iloc[-1]
        
        feats = {
            'rsi_14': last_row['rsi_14'],
            'macd_hist': last_row['macd_hist'],
            'price_vs_sma50': last_row['price_vs_sma50']
        }
        
        # One-hot encoding trick to match training features
        for c in self.feature_cols:
            if c.startswith('pattern_'):
                feats[c] = 1 if c == f'pattern_{pattern}' else 0
            elif c not in feats:
                feats[c] = 0 # Default fallback
                
        X = pd.DataFrame([feats])[self.feature_cols]
        
        outcomes = {}
        for h, model in self.outcome_models.items():
            prob = model.predict_proba(X)[0][1] # Probability of class 1 (up)
            outcomes[f"{h}d_horizon"] = round(prob * 100, 2)
            
        return {
            "ticker": ticker,
            "pattern_detected": pattern,
            "cnn_confidence": round(cnn_confidence * 100, 2),
            "forward_returns": outcomes,
            "sample_size": "N/A", # Needs historical lookup DB
            "chart_base64": img_base64
        }
