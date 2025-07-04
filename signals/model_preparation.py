# === signals/model_preparation.py ===

import numpy as np

from signals.feature_engineer import FeatureEngineer
from signals.pattern_detector import CandlestickPatternDetector
from signals.technical_indicators import TechnicalIndicatorGenerator


def prepare_model_data(df, window=5, threshold=0.001):
    """
    Combines indicator calculation, pattern detection, and feature engineering
    with label generation for supervised model training.

    Returns:
        X: pd.DataFrame – feature matrix
        y: pd.Series – binary target (1 = up, 0 = down)
        feature_names: list of selected feature column names
    """
    # === Step 1: Compute indicators, patterns, features ===
    df = TechnicalIndicatorGenerator(df).compute()
    df = CandlestickPatternDetector(df).detect()
    df = FeatureEngineer(df).engineer()

    # === Step 2: Create target label based on forward return ===
    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    volatility = df['ATR'] / df['close']
    adjusted_threshold = threshold + 0.5 * volatility

    df['target'] = np.select(
        [df['return'] >= adjusted_threshold, df['return'] <= -adjusted_threshold],
        [1, 0], default=-1
    )

    # Filter out ambiguous samples (label = -1)
    df = df[df['target'] != -1].copy()

    # === Step 3: Extract features and target ===
    y = df['target']
    X = df.drop(columns=['next_close', 'return', 'target'], errors='ignore')

    valid_idx = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    feature_names = list(X.columns)

    # === Step 4: Print summary info ===
    print(f"[DataPrep] Final dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"[DataPrep] Class balance: ↑ {sum(y==1)} | ↓ {sum(y==0)}")

    return X, y, feature_names

def prepare_volatility_data(df, future_window=45, threshold=0.003):
    """
    Prepares input features and labels for volatility prediction.
    Applies full indicator, pattern, and feature engineering steps,
    then constructs volatility labels and filters out NaNs.

    Returns:
        X: pd.DataFrame – feature matrix
        y: pd.Series – binary target
        feature_names: list – columns used in X
    """
    # === Step 1: Apply all transformations ===
    df = TechnicalIndicatorGenerator(df).compute()
    df = CandlestickPatternDetector(df).detect()
    df = FeatureEngineer(df).engineer()

    # === Step 2: Drop rows with incomplete features (esp. due to rolling windows) ===
    df.dropna(inplace=True)

    # === Step 3: Generate volatility target ===
    df['future_high'] = df['high'].rolling(window=future_window).max().shift(-future_window)
    df['future_low'] = df['low'].rolling(window=future_window).min().shift(-future_window)
    df['future_range'] = (df['future_high'] - df['future_low']) / df['close']
    df['vol_target'] = (df['future_range'] > threshold).astype(int)

    # === Step 4: Drop helper columns ===
    drop_cols = ['future_high', 'future_low', 'future_range']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # === Step 5: Final filter for valid samples ===
    y = df['vol_target']
    X = df.drop(columns=['vol_target'], errors='ignore')

    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]

    print(f"[VolPrep] Final volatility dataset: {len(X)} samples, {len(X.columns)} features")

    return X, y, list(X.columns)

def prepare_features_only(df):
    """
    Applies full transformation: indicators + patterns + feature engineering.
    Used in live prediction to match training structure.
    """
    try:
        df = TechnicalIndicatorGenerator(df).compute()
        df = CandlestickPatternDetector(df).detect()
        df = FeatureEngineer(df).engineer()

        return df

    except Exception as e:
        print("[Prep][Error] Feature preparation failed:", e)
        import traceback
        traceback.print_exc()
        return df