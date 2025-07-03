import talib
import numpy as np

class CandlestickPatternDetector:
    """
    Detects candlestick patterns using TA-Lib and generates associated trading signals.
    """
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
        self.bullish_patterns = ['Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
                                 'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike']
        self.bearish_patterns = ['HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
                                 'EveningDojiStar', 'EveningStar', 'GravestoneDoji']

    def detect(self):
        df = self.df

        for name, func in self.patterns.items():
            df[name] = func(df['open'], df['high'], df['low'], df['close'])

        for name in self.patterns:
            col = f"Signal_{name}"
            if name == "GravestoneDoji":
                df[col] = df[name].apply(lambda x: -1 if x == 100 else 0)
            elif name in self.bullish_patterns:
                df[col] = df[name].apply(lambda x: 1 if x == 100 else 0)
            elif name in self.bearish_patterns:
                df[col] = df[name].apply(lambda x: -1 if x == -100 else 0)
            else:
                df[col] = df[name].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        signal_cols = [f"Signal_{name}" for name in self.patterns.keys()]
        df['total_bullish_signals'] = df[signal_cols].apply(lambda row: sum(x == 1 for x in row), axis=1)
        df['total_bearish_signals'] = df[signal_cols].apply(lambda row: sum(x == -1 for x in row), axis=1)
        df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

        return df
