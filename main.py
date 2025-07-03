import atexit
import time
import joblib
import os
import pandas as pd
from binance.client import Client
from config.load_env import load_keys
from logs.signal_logger import SignalHistoryLogger
from logs.prediction_logger import PredictionLogger
from signals.indicator_module import TechnicalIndicatorGenerator, FeatureEngineer
from signals.pattern_detector import CandlestickPatternDetector
from signals.model import SignalModel
from signals.vol_model import VolatilityModel
from realtime.plot_signals import plot_realtime_signals
from validation.bootstrap import bootstrap_accuracy_pvalue
from data.data_fetcher import fetch_binance_data


def update_signal(signal_logger, signal_type, timestamp, price, confidence_str):
    signal_logger.remove_opposite_signal(timestamp, signal_type)
    if not signal_logger.has_signal(timestamp, signal_type):
        signal_logger.add_signal(signal_type, timestamp, price, trigger=confidence_str)


def run_live_loop():
    # Load API keys
    keys = load_keys()
    client = Client(keys['api_key'], keys['secret_key'])
    symbol = 'BTCUSDT'

    # Load models (ensure compatible sklearn/xgboost versions)
    if not os.path.exists('models/volatility_model.pkl'):
        raise FileNotFoundError("Missing 'models/volatility_model.pkl'. Re-train or check path.")
    if not os.path.exists('models/volatility_features.pkl'):
        raise FileNotFoundError("Missing 'models/volatility_features.pkl'. Re-train or check path.")

    vol_model = joblib.load('models/volatility_model.pkl')
    vol_features = joblib.load('models/volatility_features.pkl')

    try:
        signal_model = SignalModel(pd.DataFrame()).load()
    except Exception as e:
        raise RuntimeError("Error loading XGBoost signal model. Ensure it was saved with compatible XGBoost and scikit-learn versions.") from e

    # Set up loggers
    signal_logger = SignalHistoryLogger('data/signal_history.csv')
    prediction_logger = PredictionLogger()
    atexit.register(signal_logger.save_to_csv)
    atexit.register(prediction_logger.save_to_csv)

    print("Starting Real-Time Signal Loop...")

    while True:
        try:
            df = fetch_binance_data(symbol=symbol, interval='1m', limit=1000)
            df = TechnicalIndicatorGenerator(df).compute()
            df = CandlestickPatternDetector(df).detect()
            df = FeatureEngineer(df).engineer()

            latest = df.iloc[[-2]].copy()  # use second-to-last candle

            # Volatility Check
            vol_input = latest[vol_features]
            if vol_input.isnull().any().any():
                print("NaNs in volatility features. Skipping.")
                time.sleep(60)
                continue

            is_active = vol_model.predict(vol_input)[0] == 1
            prediction = "NEUTRAL"

            if is_active:
                X = latest[signal_model.selected_features]
                if X.isnull().any().any():
                    print("NaNs in signal features. Skipping.")
                    time.sleep(60)
                    continue

                prob = signal_model.pipeline.predict_proba(X)[0][1]

                if prob >= 0.85:
                    prediction = "UP"
                    update_signal(signal_logger, 'xgboost_bullish', latest.index[0], latest['close'].iloc[0], f"Conf={prob:.2%}")
                    print(f"XGBoost: UP (conf: {prob:.2%})")
                elif prob <= 0.15:
                    prediction = "DOWN"
                    update_signal(signal_logger, 'xgboost_bearish', latest.index[0], latest['close'].iloc[0], f"Conf={1 - prob:.2%}")
                    print(f"XGBoost: DOWN (conf: {1 - prob:.2%})")
                else:
                    print(f"Confidence ({prob:.2%}) not strong enough. Hold.")
            else:
                print("Market not volatile. No action.")

            # Evaluation
            ts = df.index[-1]
            close_now = df['close'].iloc[-2]
            close_next = df['close'].iloc[-1]
            prediction_logger.record_prediction(ts, prediction, close_next, close_now)
            print(f"Hit rate: {prediction_logger.get_hit_rate():.2%}")

            df_pred = prediction_logger.to_dataframe()
            if len(df_pred) >= 30:
                pval, acc, boot_mean = bootstrap_accuracy_pvalue(df_pred['hit'].values)
                print(f"Significance: acc={acc:.2%}, p={pval:.4f} | baseline={boot_mean:.2%}")

            plot_realtime_signals(df, symbol=symbol, signal_df=signal_logger.get_history())

            print("Sleeping for 60 seconds...\n")
            time.sleep(60)

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error in live loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    run_live_loop()