# CandleSense — Candlestick Pattern Recognition & Outcome Prediction

## Project Summary
CandleSense is an ML system that pulls historical stock data via `yfinance`, renders rolling-window candlestick charts, classifies known candlestick patterns (Doji, Hammer, Bullish Engulfing, Bearish Engulfing, Shooting Star) using a CNN, and estimates the historical forward-return probability associated with each detected pattern for a given stock. The system is exposed through a lightweight web UI so retail users — particularly those investing small amounts — can look up a ticker, see the currently forming pattern, and see backtested statistics (not guarantees) on what tends to happen next.

**Target audience**: Retail investors with small capital who want data-informed signals rather than gut-feel trading, delivered in plain language with transparent confidence/sample-size disclosure.

**Core design principle**: This is two separate models, not one.
1. **Pattern Classifier** — image → pattern label (high accuracy achievable, well-defined visual task).
2. **Outcome Estimator** — pattern + market context → forward-return probability (a statistical edge estimate, backtested, shown with confidence intervals and sample sizes — never presented as a guarantee).

This separation matters both technically (different training data, different evaluation metrics) and ethically (mixing them risks users reading "58% confidence" as investment certainty).

---

## Phase 0 — Environment & Requirements

### Core dependencies
```
python>=3.10
yfinance
pandas
numpy
TA-Lib          # for auto-labeling patterns from OHLC data
mplfinance      # for rendering candlestick chart images
scikit-learn
xgboost
torch / torchvision   # or tensorflow — pick one, torch recommended
backtrader              # or vectorbt — for backtesting
fastapi
uvicorn
streamlit               # MVP UI
pillow
matplotlib
python-dotenv
```

Note: `TA-Lib` requires the underlying C library installed separately (not just pip). On most systems: install the TA-Lib C library first, then `pip install TA-Lib`. If this proves painful in the build environment, fall back to a pure-Python reimplementation of the 5 target pattern rules (documented in Phase 2) — this avoids a fragile dependency.

### Project structure
```
candlesense/
├── data/
│   ├── raw/                  # cached OHLCV parquet files per ticker
│   ├── labeled/               # windows + pattern labels (csv/parquet)
│   └── images/                 # generated candlestick PNGs, organized by class
│       ├── doji/
│       ├── hammer/
│       ├── bullish_engulfing/
│       ├── bearish_engulfing/
│       └── shooting_star/
├── src/
│   ├── ingest.py              # yfinance data pulling + caching
│   ├── labeling.py            # TA-Lib / rule-based pattern labeling
│   ├── chart_gen.py           # mplfinance image rendering
│   ├── dataset.py             # PyTorch Dataset/DataLoader for images
│   ├── baseline_model.py      # XGBoost feature-based classifier (sanity check)
│   ├── cnn_model.py           # CNN architecture + training loop
│   ├── outcome_model.py       # forward-return probability model
│   ├── backtest.py            # backtrader/vectorbt strategy simulation
│   └── inference.py           # end-to-end: ticker -> chart -> pattern -> stats
├── api/
│   └── main.py                # FastAPI endpoints wrapping inference.py
├── ui/
│   └── app.py                 # Streamlit front end
├── notebooks/
│   └── eda.ipynb              # exploratory analysis, class balance checks
├── models/                    # saved model weights (.pt / .json)
├── tests/
├── requirements.txt
├── config.yaml                # tickers, date ranges, window size, thresholds
└── README.md
```

---

## Phase 1 — Data Ingestion

**File: `src/ingest.py`**

- Pull daily OHLCV data via `yfinance.download()` for a configurable list of tickers (start with 20–30 liquid, well-known stocks across sectors to get diverse pattern examples — e.g., large-caps for stability, a few volatile mid-caps for sharper patterns).
- Cache each ticker's data as Parquet in `data/raw/` keyed by ticker + date range, so repeated experiments don't re-hit the API.
- Store: `Open, High, Low, Close, Volume, Date`.
- Handle missing/NaN rows (holidays, splits) — forward-fill or drop, document choice.
- Config-driven: `config.yaml` holds ticker list, date range, and candle interval (start with daily; intraday is a stretch goal — see Phase 7).

**Acceptance check**: Running `ingest.py` produces one clean Parquet file per ticker with no NaNs in OHLCV columns and validated Date monotonicity.

---

## Phase 2 — Auto-Labeling Patterns

**File: `src/labeling.py`**

Use TA-Lib's candlestick recognition functions to auto-label historical candles — this avoids manual labeling of thousands of images:

| Pattern | TA-Lib function |
|---|---|
| Doji | `CDLDOJI` |
| Hammer | `CDLHAMMER` |
| Bullish Engulfing | `CDLENGULFING` (positive value) |
| Bearish Engulfing | `CDLENGULFING` (negative value) |
| Shooting Star | `CDLSHOOTINGSTAR` |

- TA-Lib returns +100 / -100 / 0 per row — convert to categorical labels.
- For each detected pattern instance, extract a **rolling window** of the preceding N candles (start with N=15–20) ending at the pattern candle — this window is what gets rendered as the image, giving the CNN market context, not just an isolated candle.
- **Class balance check is mandatory here**: Doji and Hammer will vastly outnumber clean Engulfing/Shooting Star instances. Log class counts. If imbalance exceeds ~5:1, apply either undersampling of majority classes or augmentation (slight window-length jitter, different tickers) for minority classes — do this before touching the model, not after seeing poor recall.
- Save the labeled window index (ticker, start_date, end_date, label) to `data/labeled/windows.parquet`. Do not regenerate images if this index already exists for a given config — image generation is the expensive step.

**Acceptance check**: A random sample of 20 labeled windows, manually eyeballed against their rendered charts, should visually match the stated pattern. If TA-Lib mislabels obviously (it can be noisy on Doji thresholds), tune the `penetration` parameter or add a manual rule-based filter on top.

---

## Phase 3 — Chart Image Generation

**File: `src/chart_gen.py`**

- Use `mplfinance` to render each labeled window as a clean candlestick PNG.
- **Strip all non-essential visual noise**: no axis labels, no legend, no title, no gridlines, no volume subplot (unless you deliberately want the model to learn from volume shape too — if so, keep it consistent across all classes).
- Fixed image size (e.g., 224x224, matching common CNN input dims) and fixed color scheme (standard green/red or up/down colors) across the entire dataset — consistency here is what lets the CNN learn shape rather than incidental styling artifacts.
- Save to `data/images/<class_name>/<ticker>_<date>.png`.
- Log generation progress and final per-class image counts.

**Acceptance check**: Spot-check 5 images per class visually. Confirm file counts match the labeled window index.

---

## Phase 4 — Baseline Model (do this before the CNN)

**File: `src/baseline_model.py`**

Before investing time in the CNN pipeline, train a fast tabular baseline to sanity-check that these patterns carry learnable signal at all:

- Engineer features directly from OHLC values per window: candle body size, upper/lower wick ratios, body position relative to prior candle, relative volume, N-candle trend direction.
- Train `XGBoost`/`LightGBM` multiclass classifier on these features → pattern label.
- Evaluate with accuracy, per-class precision/recall/F1, and confusion matrix (patterns most likely to be confused: Doji vs small-body Hammer, Bullish vs Bearish Engulfing if features aren't directional enough).

This baseline typically trains in seconds and gives you a benchmark the CNN must beat to justify the added complexity. It may also outperform the CNN outright — that's a valid and useful outcome to report.

**Acceptance check**: Baseline accuracy documented and confusion matrix reviewed. This becomes the "beat this number" bar for Phase 5.

---

## Phase 5 — CNN Pattern Classifier

**File: `src/cnn_model.py`**

- Architecture: transfer learning with **ResNet18** or **EfficientNet-B0** (pretrained on ImageNet), fine-tune final layers on the 5-class candlestick dataset. These are small enough to train on a single GPU or even CPU in reasonable time — no need for anything larger given the dataset size and task simplicity.
- Standard image augmentation: none or minimal (candlestick shape is meaningful — avoid rotation/flip augmentations that would invert bullish/bearish meaning; horizontal flip especially must be avoided since it reverses time order).
- Train/val/test split by **ticker and date range**, not randomly — random splitting risks leaking near-duplicate overlapping windows between train and test (since windows are rolling, adjacent windows share most of their candles). Split by time (e.g., train on 2018–2023, test on 2024–2025) or by holding out entire tickers.
- Loss: cross-entropy with class weighting to counter any remaining imbalance from Phase 2.
- Track: per-class precision/recall/F1, confusion matrix, and compare directly against the Phase 4 baseline.

**Acceptance check**: CNN test performance ≥ baseline, with particular attention to recall on minority classes (Engulfing patterns, Shooting Star) — these are rarer but often the more actionable signals.

---

## Phase 6 — Outcome / Forward-Return Model

**File: `src/outcome_model.py`**

This is the statistically honest core of the project — treat it as an edge estimator, not a prediction oracle.

- For every labeled pattern instance in the dataset, compute forward return over multiple horizons (e.g., +1, +3, +5 trading days) relative to the pattern's closing price.
- Add contextual features beyond the pattern itself: prevailing trend (e.g., price relative to 20/50-day moving average), RSI, MACD signal, relative volume — pattern-only predictions are known in finance literature to be weak; context materially improves signal.
- Train a secondary classifier (logistic regression as an interpretable baseline, gradient boosting for performance) predicting P(price up | pattern + context) per horizon.
- **Report everything with sample size.** A pattern's historical win rate is meaningless without n. If a pattern occurred fewer than ~30 times for a given ticker, flag it as low-confidence in the UI rather than showing a bare percentage.
- Calibrate probability outputs (e.g., Platt scaling) so a stated "62% confidence" actually corresponds to ~62% empirical frequency in held-out data — uncalibrated model probabilities are commonly misleading.

**Acceptance check**: Reliability diagram (calibration curve) reviewed; win-rate stats per pattern per horizon documented with sample sizes.

---

## Phase 7 — Backtesting

**File: `src/backtest.py`**

- Use `backtrader` or `vectorbt` to simulate a simple strategy: enter a position when a pattern is detected with outcome-model confidence above a threshold, exit after the corresponding horizon.
- Report: cumulative return, Sharpe ratio, max drawdown, win rate, number of trades — across multiple tickers and time periods (avoid backtesting only on the same window used for training the outcome model — use out-of-sample periods).
- This step is what determines whether the project has any real value to show users — if backtested performance doesn't clear a naive buy-and-hold or random-entry baseline over the same period, that's important to know and disclose, not hide.

**Acceptance check**: Backtest report comparing strategy returns vs. buy-and-hold baseline over the same period, for at least 5 different tickers.

---

## Phase 8 — Inference Pipeline

**File: `src/inference.py`**

End-to-end function: given a ticker string →
1. Pull latest N candles via `yfinance`.
2. Render the current window as a chart image (reuse `chart_gen.py`).
3. Run the CNN classifier → detected pattern + confidence.
4. Run the outcome model → forward-return probability + horizon + sample size + calibration-adjusted confidence.
5. Return a structured response (JSON-serializable) with all of the above plus the chart image path.

This is the function both the API and UI will call.

---

## Phase 9 — API Layer

**File: `api/main.py`** (FastAPI)

Endpoints:
- `GET /predict?ticker=AAPL` → returns JSON: detected pattern, confidence, forward-return probability per horizon, sample size, chart image (base64 or URL).
- `GET /backtest?ticker=AAPL&pattern=hammer` → returns historical backtest stats for that specific pattern/ticker combination.
- `GET /health` → basic liveness check.

Keep this stateless and cache recent ticker pulls briefly (e.g., 15 min TTL) to avoid hammering the yfinance API on repeated requests for the same ticker.

---

## Phase 10 — User Interface (MVP: Streamlit)

**File: `ui/app.py`**

- Ticker search input.
- Rendered candlestick chart with the detected pattern region highlighted/annotated.
- Plain-language card, e.g.:
  > **Pattern detected: Bullish Engulfing** (92% classifier confidence)
  > Historically, over the next 5 trading days, this stock rose in **58% of 41 similar past occurrences**.
  > *Sample size is moderate — treat as directional signal, not certainty.*
- Always show sample size prominently — this is a hard requirement given the audience is small-capital retail investors who may over-trust a bare percentage.
- Prominent, persistent disclaimer: **"This tool provides statistical/educational information based on historical patterns. It is not financial advice. Past performance does not guarantee future results."**
- Optional: a small backtest chart showing equity curve for that pattern on that ticker, so users can visually judge historical reliability rather than trusting a single number.

**Stretch goal for later**: migrate from Streamlit MVP to a React/Next.js front end once the model pipeline is validated, for a more polished, mobile-friendly experience suited to the target audience.

---

## Build Order (do in this sequence)

1. `ingest.py` — data pipeline, validate clean OHLCV output.
2. `labeling.py` — auto-label via TA-Lib, manually spot-check accuracy.
3. `chart_gen.py` — generate images, verify class balance and visual correctness.
4. `baseline_model.py` — XGBoost sanity check; confirms patterns carry signal before investing in CNN work.
5. `cnn_model.py` — train CNN, benchmark against baseline, focus on minority-class recall.
6. `outcome_model.py` — forward-return probability model with calibration.
7. `backtest.py` — validate whether the whole system has real historical edge vs. buy-and-hold.
8. `inference.py` — wire it all into one callable pipeline.
9. `api/main.py` — expose via FastAPI.
10. `ui/app.py` — Streamlit front end for non-technical users.

Do not skip Phase 4 (baseline) or Phase 7 (backtest) to save time — these are the two steps that tell you honestly whether this project works, and both are cheap relative to the CNN training effort.

---

## Known Risks & Design Constraints to Respect

- **Data leakage via rolling windows**: overlapping windows near class boundaries can leak between train/test splits if split randomly. Always split by time or by ticker, never by random row shuffling.
- **Horizontal flip augmentation is invalid** for candlestick images — it reverses temporal order and would corrupt bullish/bearish meaning.
- **Pattern-only signal is weak** in finance literature — the outcome model must include broader market context (trend, RSI, volume), not just the pattern label, or it will underperform and mislead users.
- **Class imbalance** is expected and must be measured and addressed explicitly (Doji/Hammer >> Engulfing/Shooting Star in raw frequency).
- **Always report sample size alongside any probability** shown to end users — this is a non-negotiable UI requirement for the target audience of small-capital retail investors.
- **TA-Lib installation friction**: the C library dependency can be a build blocker in some environments — have the rule-based fallback ready.
- **This is not investment advice** — disclaimers must be persistent and visible in the UI, not buried in a footer.

---

## Stretch Goals (post-MVP)
- Expand beyond the initial 5 patterns to the full TA-Lib pattern set (60+ patterns).
- Intraday timeframes (1h/15m) in addition to daily.
- Multi-pattern sequence modeling (e.g., LSTM/Transformer over sequences of detected patterns rather than single-pattern snapshots).
- Portfolio-level view: scan a watchlist of tickers and surface all currently-forming patterns at once.
- User accounts + notification when a tracked ticker forms a high-confidence pattern.
