from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.inference import InferencePipeline

app = Flask(__name__)
CORS(app)

pipeline = None

def load_models():
    global pipeline
    try:
        pipeline = InferencePipeline()
    except Exception as e:
        print(f"Warning: Could not initialize pipeline. Ensure models exist. Error: {e}")

# Load models at startup
load_models()

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["GET"])
def predict():
    if not pipeline:
        return jsonify({"error": "Models not loaded"}), 500
        
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400
        
    result = pipeline.predict(ticker.upper())
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
        
    return jsonify(result)

@app.route("/backtest", methods=["GET"])
def backtest():
    ticker = request.args.get("ticker")
    pattern = request.args.get("pattern")
    
    if not ticker or not pattern:
        return jsonify({"error": "Ticker and pattern parameters are required"}), 400
        
    # Placeholder for backtest stats
    return jsonify({
        "ticker": ticker,
        "pattern": pattern,
        "historical_win_rate": 55.2,
        "sample_size": 42
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
