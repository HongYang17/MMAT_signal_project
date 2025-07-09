# QF635 Algorithmic Trading Project — BTCUSDT Real-Time Signal Engine

This repository implements an end-to-end crypto signal prediction and trading system for BTC/USDT. It integrates rule-based strategies, technical indicators, and supervised machine learning to generate directional trading signals in real time.

Developed as part of the **QF635: Market Microstructure and Algorithmic Trading** course (MQF 2024), the project focuses on building predictive trading signals using both interpretable patterns and statistical learning.

##  Environment Setup

To run this project, choose **one** of the following setup methods:

### Option 1: Conda Environment (Recommended for Windows + TA-Lib)

This method is ideal for full compatibility with **TA-Lib** and better dependency management.

```
conda env create -f environment.yml
conda activate mmat-env
```

**Note:** On Windows, `ta-lib` may not install automatically via `pip`.  
To manually install it:

1. Download the appropriate `.whl` file from:  
   👉 https://github.com/mrjbq7/ta-lib/releases

   *(Example for Python 3.10: `ta_lib-0.6.4-cp310-cp310-win_amd64.whl`)*

2. Install it after activating the environment:

```
pip install wheels/ta_lib-*.whl
```

3. Verify it worked:

```
python -c "import talib; print(talib.__version__)"
```



###  Option 2: Virtualenv + pip (Linux/macOS or minimal setups)

If you're not using Conda, create and activate a virtual environment:

```bash
python -m venv venv

# Activate the environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Then install core dependencies:
pip install -r requirements.txt
```

**Important:**  
TA-Lib must still be installed manually via `.whl` (Windows) or system libraries (Linux/macOS).  
See `requirements.txt` for installation tips.



## Contributors & Module Responsibilities

| Team Member            | Role & Focus Area                                          | Code Location                                               | Personal Repo|
|------------------------|------------------------------------------------------------|--------------------------------------------------------------|----|
| **Chang Wen Yu**       | Candlestick pattern signal engine + technical indicator fusion  | `experiments/Candlestick_withIndicators_5Min_LiveAPI.ipynb` |---|
| **Xie Zuoyu**          | Volume-based strategy design and evaluation | `experiments/signal_volume/` |https://github.com/Xie426/MM_Signal_Strategy_XIE|
| **Hong Yang**          | Meta-learning pipeline (XGBoost/GBM/Stacking) | `experiments/metamodel.ipynb`|---|
| **Li Sinuan**          | Trend indicator modeling + live XGBoost signal generator | `main.py`|https://github.com/sinuan/project4-btc-market-predict|
| **Phoo Pyae Hsu Myat** | Regime detection, confidence filters, risk-adaptive backtest | `experiments/regime_detection.ipynb`|---|



##  System Overview

###  Part 1: Candlestick + Indicator Strategy (Wenyu)
- 19 TA-Lib patterns resampled to 15-min
- Signal strength levels (strong/moderate) with trigger filters
- Indicator confirmation: MA crossover, RSI, MACD, ATR
- Live visualization from Binance 5-min API

###  Part 2: Volume-Based Signal Module (Zuoyu)
- Built OBV, AD, MFI, ADOSC, BOP signals (3-min intervals)
- Strict/relaxed signal pairing and cumulative PnL evaluation
- Strategies 2+4 and 2+5 selected for live deployment

###  Part 3: Meta-Learning Framework (Hong Yang)
- Engineered 25+ TA/volatility/volume/time features
- Built `EnhancedMetaModel` with PCA + ElasticNet
- SMOTE label balancing and Sharpe-weighted aggregation
- Validated with expanding-window backtests and White's Reality Check

###  Part 4: Trend-Signal XGBoost Classifier (Sinuan)
- 9 strategy-derived features: ADX, DI, EMA, SMA crossovers
- Live signal gating with confidence threshold (p > 0.85)
- Volatility classifier disables signal in flat conditions
- Real-time logic implemented in `main.py`

###  Part 5: Regime Segmentation and Risk Filters (Myat)
- Sliding 12-hour regime labeling: HighVol/LowVol Trends + Sideways
- Integrated confidence threshold + volatility filter
- Conducted regime-specific evaluation
- Interactive Plotly viewer with `ipywidgets`



##  Evaluation Summary

- **Backtest Sharpe (Meta-Model)**: 4.70 (weighted ensemble)
- **Hit Rate**: 59% in LowVol_DownTrend regime
- **Real-Time Accuracy (XGB)**: ~51% after 6-hour runtime
- **Volatility Filter Impact**: Improved precision, reduced noise
- **White’s Reality Check**: p ≈ 0.486 → borderline statistical confidence

---


