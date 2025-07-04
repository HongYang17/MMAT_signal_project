# === signals/pattern_detector.py ===

import traceback

import numpy as np
import pandas as pd
import talib


class CandlestickPatternDetector:
    def __init__(self, df):
        self.df = df.copy()
        self.patterns = {
            'Hammer': talib.CDLHAMMER,
            'InvertedHammer': talib.CDLINVERTEDHAMMER,
            'BullishEngulfing': talib.CDLENGULFING,
            'PiercingLine': talib.CDLPIERCING,
            'MorningStar': talib.CDLMORNINGSTAR,
            'DragonflyDoji': talib.CDLDRAGONFLYDOJI,
            'LongLine': talib.CDLLONGLINE,
            'ThreeLineStrike': talib.CDL3LINESTRIKE,
            'HangingMan': talib.CDLHANGINGMAN,
            'ShootingStar': talib.CDLSHOOTINGSTAR,
            'BearishEngulfing': talib.CDLENGULFING,
            'DarkCloudCover': talib.CDLDARKCLOUDCOVER,
            'EveningDojiStar': talib.CDLEVENINGDOJISTAR,
            'EveningStar': talib.CDLEVENINGSTAR,
            'GravestoneDoji': talib.CDLGRAVESTONEDOJI,
        }

        self.bullish_patterns = [
            'Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
            'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike'
        ]
        self.bearish_patterns = [
            'HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
            'EveningDojiStar', 'EveningStar', 'GravestoneDoji'
        ]

    def detect(self):
        df = self.df
        failed_patterns = []

        # Step 1: Compute raw pattern signals
        for name, func in self.patterns.items():
            try:
                df[name] = func(df['open'], df['high'], df['low'], df['close'])
            except Exception as e:
                print(f"[PatternError] {name} failed: {e}")
                traceback.print_exc()
                df[name] = pd.Series(np.zeros(len(df)), index=df.index)
                failed_patterns.append(name)

        # Step 2: Convert to 1/0/-1 based Signal_ columns
        for name in self.patterns:
            col = f"Signal_{name}"
            try:
                if name == "GravestoneDoji":
                    df[col] = df[name].apply(lambda x: -1 if x == 100 else 0)
                elif name in self.bullish_patterns:
                    df[col] = df[name].apply(lambda x: 1 if x == 100 else 0)
                elif name in self.bearish_patterns:
                    df[col] = df[name].apply(lambda x: -1 if x == -100 else 0)
                else:
                    df[col] = df[name].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            except Exception as e:
                print(f"[SignalError] Failed to assign Signal_{name}: {e}")
                traceback.print_exc()
                df[col] = 0

        # Step 3: Ensure all Signal_* columns exist
        signal_cols = [f"Signal_{name}" for name in self.patterns]
        for col in signal_cols:
            if col not in df.columns:
                print(f"[Fallback] Creating missing signal column: {col}")
                df[col] = 0

        # Step 4: Add summary signal features
        df['total_bullish_signals'] = df[signal_cols].apply(lambda row: sum(x == 1 for x in row), axis=1)
        df['total_bearish_signals'] = df[signal_cols].apply(lambda row: sum(x == -1 for x in row), axis=1)
        df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

        if failed_patterns:
            print(f"[Summary] Patterns failed to compute: {', '.join(failed_patterns)}")

        return df