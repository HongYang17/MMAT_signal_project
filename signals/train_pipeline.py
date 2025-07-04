from datetime import datetime

import pandas as pd

from data_fetching.data_fetcher import update_binance_csv
from signals.pattern_detector import CandlestickPatternDetector
from signals.technical_indicators import TechnicalIndicatorGenerator, FeatureEngineer
from train_model.signal_model import SignalModel
from train_model.vol_model import VolatilityModel
from validation.metrics import evaluate_model
from validation.pattern_evaluator import evaluate_patterns, summarize_pattern_performance


def run_pipeline_on_df(df, evaluate=True):
    """
    Run full pipeline: feature engineering, volatility filtering, signal modeling, evaluation.
    Returns trained signal model, volatility model, and enriched DataFrame.
    """
    # === Step 1: Generate technical indicators & engineered features ===
    df = TechnicalIndicatorGenerator(df).compute()
    df = CandlestickPatternDetector(df).detect()
    df = FeatureEngineer(df).engineer()

    # === Step 2: Volatility modeling ===
    vol_model = VolatilityModel(df)
    df = vol_model.label_future_volatility()
    vol_model.train()
    vol_model.save()

    # === Step 3: Signal modeling ===
    df['next_close'] = df['close'].shift(-5)
    df['return'] = (df['next_close'] - df['close']) / df['close']
    df['target'] = (df['return'] > 0.001).astype(int)

    signal_model = SignalModel(df)
    pipeline, features = signal_model.train()
    signal_model.save()

    # === Step 4: Evaluate model and pattern accuracy ===
    if evaluate:
        X_eval, y_eval = signal_model.prepare_data(features)
        evaluate_model(pipeline, X_eval, y_eval, title="Signal Model")

        pattern_results = evaluate_patterns(df, CandlestickPatternDetector(df).patterns)
        summary = summarize_pattern_performance(pattern_results)
        print("\nTop Pattern Accuracies:")
        print(summary.head(10))

    return signal_model, vol_model, df


def run_training_pipeline(symbol="BTCUSDT", csv_path=None):
    """
    Entry point for running pipeline on live or local data.
    """
    # === Step 0: Load data (CSV or online) ===
    if csv_path:
        print(f"\n[INFO] Loading historical data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        filename = f"data/{symbol}_1min_{datetime.today().date()}.csv"
        print(f"\n[INFO] Downloading latest data to {filename}")
        update_binance_csv(symbol=symbol, file_path=filename)
        df = pd.read_csv(filename, parse_dates=["timestamp"], index_col="timestamp")

    # === Step 1–4: Run full pipeline ===
    run_pipeline_on_df(df)


if __name__ == "__main__":
    run_training_pipeline()