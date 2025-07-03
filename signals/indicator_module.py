# === signals/indicator_module.py ===
import talib

class TechnicalIndicatorGenerator:
    def __init__(self, df):
        self.df = df.copy()

    def compute(self):
        df = self.df
        df = self._add_momentum_indicators(df)
        df = self._add_trend_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_volatility_indicators(df)
        df.dropna(inplace=True)
        return df

    def _add_momentum_indicators(self, df):
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], 12, 26, 9)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MOM'] = talib.MOM(df['close'], timeperiod=10)
        return df

    def _add_trend_indicators(self, df):
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
        df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['EMA200'] = talib.EMA(df['close'], timeperiod=200)
        return df

    def _add_volume_indicators(self, df):
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        df['Volume_MA'] = talib.SMA(df['volume'], timeperiod=20)
        return df

    def _add_volatility_indicators(self, df):
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)
        df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=20)
        df['MA5'] = talib.SMA(df['close'], timeperiod=5)
        df['MA20'] = talib.SMA(df['close'], timeperiod=20)
        df['Volume_MA20'] = talib.SMA(df['volume'], timeperiod=20)
        df['mean_ATR'] = df['ATR'].rolling(20).mean()
        return df


class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def engineer(self):
        df = self.df
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['close']
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volatility_ratio'] = df['ATR'] / df['close'].rolling(20).mean().shift(1)
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ratio'] = df['volume'] / df['Volume_MA']
        df['rsi_divergence'] = df['RSI'] - df['RSI'].rolling(5).mean().shift(1)
        df['macd_hist'] = df['MACD'] - df['MACD_signal']
        df['distance_to_upper_bb'] = (df['Upper_BB'] - df['close']) / df['close']
        df['distance_to_lower_bb'] = (df['close'] - df['Lower_BB']) / df['close']
        df['trend_power'] = df['ADX'] * (df['PLUS_DI'] - df['MINUS_DI'])

        for col in ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']:
            for lag in [1, 2, 3, 5, 10]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        for col in ['RSI', 'MACD', 'ATR', 'volume', 'close']:
            df[f'{col}_pct_change'] = df[col].pct_change()

        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            df['macd_histogram'] = df['MACD'] - df['MACD_signal']
        else:
            df['macd_histogram'] = 0.0  # fallback to neutral value or np.nan

        df['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)

        df.dropna(inplace=True)
        return df
