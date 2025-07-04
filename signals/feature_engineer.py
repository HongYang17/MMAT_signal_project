# === signals/feature_engineer.py ===

import traceback

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def engineer(self):
        df = self.df
        errors = []

        try:
            # === Volume_MA fallback ===
            if 'Volume_MA' not in df.columns:
                df['Volume_MA'] = df['volume'].rolling(20).mean()

            # === Price-based features ===
            df['close_to_high'] = (df['high'] - df['close']) / df['high']
            df['close_to_low'] = (df['close'] - df['low']) / df['close']
            df['price_range'] = (df['high'] - df['low']) / df['close']

            # === Volatility and trend indicators ===
            df['volatility_ratio'] = df['ATR'] / df['close'].rolling(20).mean().shift(1)
            df['trend_power'] = df['ADX'] * (df['PLUS_DI'] - df['MINUS_DI'])
            df['distance_to_upper_bb'] = (df['Upper_BB'] - df['close']) / df['close']
            df['distance_to_lower_bb'] = (df['close'] - df['Lower_BB']) / df['close']

            # === Momentum and volume behaviors ===
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ratio'] = df['volume'] / df['Volume_MA']
            df['rsi_divergence'] = df['RSI'] - df['RSI'].rolling(5).mean().shift(1)
            df['macd_histogram'] = df['MACD'] - df['MACD_signal']
            df['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)

        except Exception as e:
            print("[FeatureEngineer][Error] Failed during main feature computation:", e)
            traceback.print_exc()
            errors.append("main features")

        try:
            # === Lag features ===
            lag_targets = ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']
            for col in lag_targets:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        except Exception as e:
            print("[FeatureEngineer][Error] Failed during lag features:", e)
            traceback.print_exc()
            errors.append("lag features")

        try:
            # === Percentage change features ===
            pct_targets = ['RSI', 'MACD', 'ATR', 'volume', 'close']
            for col in pct_targets:
                df[f'{col}_pct_change'] = df[col].pct_change()
        except Exception as e:
            print("[FeatureEngineer][Error] Failed during pct_change features:", e)
            traceback.print_exc()
            errors.append("pct_change")

        try:
            # === Additional fallback lag columns ===
            for col in ['ATR', 'ADX', 'close']:
                for lag in [2, 5, 10]:
                    lag_col = f'{col}_lag{lag}'
                    if lag_col not in df.columns:
                        df[lag_col] = df[col].shift(lag)
        except Exception as e:
            print("[FeatureEngineer][Warning] Failed to fill fallback lags:", e)
            traceback.print_exc()
            errors.append("fallback lags")

        try:
            # === Candlestick fallback columns (totals only) ===
            fallback_totals = ['net_candle_signal', 'total_bullish_signals', 'total_bearish_signals']
            for col in fallback_totals:
                if col not in df.columns:
                    print(f"[FeatureEngineer][Fallback] Column {col} not found, defaulting to 0")
                    df[col] = 0
        except Exception as e:
            print("[FeatureEngineer][Warning] Failed to inject candlestick fallback columns:", e)
            traceback.print_exc()
            errors.append("candlestick totals fallback")

        df.dropna(inplace=True)

        if errors:
            print(f"[FeatureEngineer][Summary] Feature engineering completed with issues: {errors}")

        return df