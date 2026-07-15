# CandleSense Implementation Walkthrough

The CandleSense AI modeling project has been successfully set up! Given the large scope described in the 10-phase plan, I have built out the **entire foundational architecture and code for all 10 phases**.

## What Was Completed

1. **Developer Documentation**
   - Created `DEVELOPER_GUIDE.md` detailing the system architecture, file purposes, and tips for extending/tweaking the models.
2. **Project Scaffolding**
   - Created `config.yaml` containing settings for tickers, training date ranges, moving averages, and model architectures.
   - Created `requirements.txt` containing all necessary dependencies for ML (XGBoost, PyTorch, scikit-learn), Data processing (pandas, yfinance, TA-Lib), and serving (FastAPI, Streamlit).
3. **Phase 1-3: Data Pipeline (`src/ingest.py`, `src/labeling.py`, `src/chart_gen.py`)**
   - Wrote code to fetch historical data from `yfinance`.
   - Used `TA-Lib` to extract pattern labels (Doji, Hammer, Engulfing) and roll windows for contextual analysis.
   - Employed `mplfinance` to generate clear, borderless image charts for the CNN.
4. **Phase 4-7: Modeling (`src/baseline_model.py`, `src/cnn_model.py`, `src/dataset.py`, `src/outcome_model.py`, `src/backtest.py`)**
   - Provided baseline XGBoost model focusing on tabular features.
   - Implemented a ResNet18 fine-tuning script in PyTorch to classify the chart images.
   - Developed an Outcome Logistic Regression model that is calibrated to predict forward-returns probability using context features (MACD, RSI).
5. **Phase 8-10: Application Layer (`src/inference.py`, `api/main.py`, `ui/app.py`)**
   - Wrapped the dual-model logic into an `InferencePipeline`.
   - Wrote a FastAPI script with a `/predict` endpoint.
   - Wrote a Streamlit frontend UI for users to analyze tickers seamlessly.

## How to Run It

To execute the project, open your terminal and follow these steps sequentially:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You will also need to install the TA-Lib C-library natively on your OS).*

2. **Run Data Generation and Training** (This will take time):
   ```bash
   python src/ingest.py
   python src/labeling.py
   python src/chart_gen.py
   python src/baseline_model.py
   python src/cnn_model.py
   python src/outcome_model.py
   ```

3. **Start the API Server**:
   ```bash
   uvicorn api.main:app --reload
   ```

4. **Start the Streamlit UI** (in a separate terminal):
   ```bash
   streamlit run ui/app.py
   ```

## Final Thoughts
The architecture handles decoupling the classification logic from the statistical forward-return predictions precisely as requested. Please review the `DEVELOPER_GUIDE.md` to see further tips on model tweaking!
