import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_PATH = os.path.join(BASE_DIR, "data", "BTCUSDT_1min.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
SIGNAL_HISTORY_PATH = os.path.join(BASE_DIR, "logs", "signal_history.csv")
PREDICTION_LOG_PATH = os.path.join(BASE_DIR, "logs", "prediction_log.csv")

VOL_MODEL_FILE = os.path.join(MODEL_DIR, "volatility_model.pkl")
VOL_FEATURES_FILE = os.path.join(MODEL_DIR, "volatility_features.json")

XGB_MODEL_FILE = os.path.join(MODEL_DIR, "final_signal_model.pkl")
XGB_FEATURES_FILE = os.path.join(MODEL_DIR, "xgb_features.json")

CONFIDENCE_THRESHOLD = 0.85
SYMBOL = "BTCUSDT"
INTERVAL = "1m"