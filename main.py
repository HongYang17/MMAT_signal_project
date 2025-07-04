import time
import traceback

import os
import pandas as pd
import pytz

from datetime import datetime, timedelta, timezone
from config import config
from data_fetching.data_fetcher import update_binance_csv
from logs.prediction_logger import PredictionLogger
from logs.signal_logger import SignalHistoryLogger
from ploting.plot_signals import plot_realtime_signals
from signals.model_preparation import prepare_features_only
from signals.pattern_detector import CandlestickPatternDetector
from train_model.signal_model import SignalModel
from train_model.vol_model import VolatilityModel
from validation.bootstrap import bootstrap_accuracy_pvalue
from validation.metrics import print_classification_metrics
from validation.pattern_evaluator import evaluate_patterns, summarize_pattern_performance

def update_signal(signal_logger, signal_type, timestamp, price, confidence_str):
    try:
        signal_logger.remove_opposite_signal(timestamp, signal_type)
        if not signal_logger.has_signal(timestamp, signal_type):
            signal_logger.add_signal(signal_type, timestamp, price, trigger=confidence_str)
            signal_logger.save_to_csv()
    except Exception as e:
        print(f"[ERROR] update_signal failed: {e}")
        traceback.print_exc()

def run_live_loop(enable_plot=True):
    print("=== Real-Time Signal Loop Initialized ===")

    symbol = config.SYMBOL
    local_tz = pytz.timezone("Asia/Singapore")

    try:
        if not os.path.exists(config.DATA_PATH):
            print("[Info] Data file missing. Creating initial dataset...")
            update_binance_csv(symbol=symbol, file_path=config.DATA_PATH)
        df_train = pd.read_csv(config.DATA_PATH)
        df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], utc=True, errors="coerce")
        df_train = df_train.set_index("timestamp")
        df_train = df_train.dropna(subset=["open", "high", "low", "close", "volume"])  # Optional clean-up

        df_train = df_train.sort_index()
    except Exception as e:
        print(f"[ERROR] Failed to fetch/load data: {e}")
        traceback.print_exc()
        return

    try:
        vol_model = VolatilityModel().load()
    except Exception:
        print("[VolModel] Reload failed: retraining...")
        vol_model = VolatilityModel()
        vol_model.train_stage1(df_train)
        vol_model.retrain_stage2(df_train)
        vol_model.save()

    try:
        signal_model = SignalModel().load()
    except Exception:
        print("[SigModel] Reload failed: retraining...")
        signal_model = SignalModel()
        signal_model.train_stage1(df_train)
        signal_model.retrain_stage2(df_train)
        signal_model.save()

    vol_features = vol_model.selected_features
    signal_features = signal_model.selected_features

    signal_logger = SignalHistoryLogger()
    prediction_logger = PredictionLogger(autosave=True)

    last_processed_ts = None

    while True:
        loop_start = time.time()
        try:
            update_binance_csv(symbol=symbol, file_path=config.DATA_PATH, max_days=395)
            df_train = pd.read_csv(config.DATA_PATH, index_col="timestamp", parse_dates=True)
            df_train.index = df_train.index.tz_localize("UTC") if df_train.index.tz is None else df_train.index
            df_train = df_train.sort_index()

            df = prepare_features_only(df_train.copy())
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index

            if len(df) < 6:
                time.sleep(60)
                continue

            latest = df.iloc[[-1]].copy()
            ts = latest.index[0]
            if ts == last_processed_ts:
                elapsed = time.time() - loop_start
                time.sleep(max(0, 60 - elapsed))
                continue

            last_processed_ts = ts
            ts_local = ts.tz_convert(local_tz)

            vol_input = latest.reindex(columns=vol_features, fill_value=0.0)
            sig_input = latest.reindex(columns=signal_features, fill_value=0.0)

            try:
                pred = vol_model.final_model.predict(vol_input)[0]
                proba = vol_model.final_model.predict_proba(vol_input)[0][1]
                is_active = pred == 1

                if not is_active:
                    if proba < 0.2:
                        print(f"[Vol] {ts_local} — Rejected due to low volatility confidence (p={proba:.2%})")
                    else:
                        print(f"[Vol] {ts_local} — Rejected: Volatility model inactive despite high confidence (p={proba:.2%})")
                else:
                    print(f"[Vol] {ts_local} — Volatility prediction: {pred} (p={proba:.2%})")

            except Exception as e:
                print(f"[ERROR] Vol prediction failed: {e}")
                traceback.print_exc()
                elapsed = time.time() - loop_start
                time.sleep(max(0, 60 - elapsed))
                continue

            if is_active:
                try:
                    prob = signal_model.final_model.predict_proba(sig_input)[0][1]
                    close_now = df['close'].iloc[-6]
                    close_next = df['close'].iloc[-5]

                    if prob >= 0.85:
                        update_signal(signal_logger, 'xgboost_bullish', ts, latest['close'].iloc[0], f"Conf={prob:.2%}")
                        prediction_logger.record_prediction(ts, "UP", close_next, close_now, confidence=prob)
                    elif prob <= 0.15:
                        update_signal(signal_logger, 'xgboost_bearish', ts, latest['close'].iloc[0], f"Conf={(1 - prob):.2%}")
                        prediction_logger.record_prediction(ts, "DOWN", close_next, close_now, confidence=prob)
                    else:
                        print(f"[Signal] {ts_local} — Rejected due to low signal confidence ({prob:.2%})")

                except Exception as e:
                    print("[ERROR] Signal prediction failed.")
                    traceback.print_exc()
                    elapsed = time.time() - loop_start
                    time.sleep(max(0, 60 - elapsed))
                    continue

            print(f"[Stats] HitRate: {prediction_logger.get_hit_rate():.2%}")

            try:
                if len(prediction_logger.log) >= 30:
                    df_eval = prediction_logger.to_dataframe()
                    y_true = df_eval['hit']
                    y_pred = df_eval['prediction']
                    confidences = df_eval.get('confidence', pd.Series([0.5] * len(df_eval)))
                    print_classification_metrics(y_true, y_pred, confidences)
                    pval, acc, base = bootstrap_accuracy_pvalue(df_eval['hit'].values)
                    print(f"[Eval] Bootstrapped Accuracy: {acc:.2%} | p={pval:.4f} | Baseline: {base:.2%}")
            except Exception as e:
                print("[Eval] Failed to compute performance stats.")
                traceback.print_exc()

            try:
                patterns_dict = CandlestickPatternDetector(df).patterns
                pattern_results = evaluate_patterns(df, patterns_dict, window=5, threshold=0.001)
                summary_df = summarize_pattern_performance(pattern_results)
                print("\n[Pattern Eval] Top Candlestick Patterns:")
                print(summary_df.head(10))
            except Exception as e:
                print("[PatternEval] Failed to evaluate candlestick patterns.")
                traceback.print_exc()

            if enable_plot:
                try:
                    df_plot = df.copy()
                    df_plot.index = df_plot.index.tz_convert(local_tz)
                    plot_realtime_signals(
                        df_plot,
                        symbol=symbol,
                        signal_logger=signal_logger,
                        output_path='data/realtime_plot.html',
                        data_range=60
                    )
                except Exception as e:
                    print("[Plot] Failed to render chart.")
                    traceback.print_exc()

            elapsed = time.time() - loop_start
            time.sleep(max(0, 60 - elapsed))

        except KeyboardInterrupt:
            print("[Exit] Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] Main loop crash: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    run_live_loop()