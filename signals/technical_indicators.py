# === signals/technical_indicators.py ===

import traceback

import numpy as np
import pandas as pd
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

    def _safe_talib_call(self, func, out_cols, *args, **kwargs):
        """
        Wraps TA-Lib calls with error handling and returns fallback NaNs if needed.
        """
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                return {col: val for col, val in zip(out_cols, result)}
            else:
                return {out_cols[0]: result}
        except Exception as e:
            print(f"[TA-Lib Error] Failed on {func.__name__}: {e}")
            traceback.print_exc()
            return {col: pd.Series(np.nan, index=self.df.index) for col in out_cols}

    def _add_momentum_indicators(self, df):
        df['RSI'] = self._safe_talib_call(talib.RSI, ['RSI'], df['close'], timeperiod=14)['RSI']
        macd = self._safe_talib_call(talib.MACD, ['MACD', 'MACD_signal', '_'], df['close'], 12, 26, 9)
        df['MACD'] = macd['MACD']
        df['MACD_signal'] = macd['MACD_signal']
        stoch = self._safe_talib_call(talib.STOCH, ['STOCH_K', 'STOCH_D'], df['high'], df['low'], df['close'])
        df['STOCH_K'] = stoch['STOCH_K']
        df['STOCH_D'] = stoch['STOCH_D']
        df['CCI'] = self._safe_talib_call(talib.CCI, ['CCI'], df['high'], df['low'], df['close'], timeperiod=14)['CCI']
        df['MOM'] = self._safe_talib_call(talib.MOM, ['MOM'], df['close'], timeperiod=10)['MOM']
        return df

    def _add_trend_indicators(self, df):
        df['ADX'] = self._safe_talib_call(talib.ADX, ['ADX'], df['high'], df['low'], df['close'], timeperiod=14)['ADX']
        df['EMA20'] = self._safe_talib_call(talib.EMA, ['EMA20'], df['close'], timeperiod=20)['EMA20']
        df['SMA20'] = self._safe_talib_call(talib.SMA, ['SMA20'], df['close'], timeperiod=20)['SMA20']
        df['PLUS_DI'] = self._safe_talib_call(talib.PLUS_DI, ['PLUS_DI'], df['high'], df['low'], df['close'], timeperiod=14)['PLUS_DI']
        df['MINUS_DI'] = self._safe_talib_call(talib.MINUS_DI, ['MINUS_DI'], df['high'], df['low'], df['close'], timeperiod=14)['MINUS_DI']
        df['EMA200'] = self._safe_talib_call(talib.EMA, ['EMA200'], df['close'], timeperiod=200)['EMA200']
        return df

    def _add_volume_indicators(self, df):
        df['OBV'] = self._safe_talib_call(talib.OBV, ['OBV'], df['close'], df['volume'])['OBV']
        df['AD'] = self._safe_talib_call(talib.AD, ['AD'], df['high'], df['low'], df['close'], df['volume'])['AD']
        df['ADOSC'] = self._safe_talib_call(talib.ADOSC, ['ADOSC'], df['high'], df['low'], df['close'], df['volume'])['ADOSC']
        df['MFI'] = self._safe_talib_call(talib.MFI, ['MFI'], df['high'], df['low'], df['close'], df['volume'], timeperiod=14)['MFI']
        df['Volume_MA'] = self._safe_talib_call(talib.SMA, ['Volume_MA'], df['volume'], timeperiod=20)['Volume_MA']
        return df

    def _add_volatility_indicators(self, df):
        df['ATR'] = self._safe_talib_call(talib.ATR, ['ATR'], df['high'], df['low'], df['close'], timeperiod=14)['ATR']
        df['NATR'] = self._safe_talib_call(talib.NATR, ['NATR'], df['high'], df['low'], df['close'], timeperiod=14)['NATR']
        df['SAR'] = self._safe_talib_call(talib.SAR, ['SAR'], df['high'], df['low'], acceleration=0.02, maximum=0.2)['SAR']
        bb = self._safe_talib_call(talib.BBANDS, ['Upper_BB', 'Middle_BB', 'Lower_BB'], df['close'], timeperiod=20)
        df['Upper_BB'] = bb['Upper_BB']
        df['Middle_BB'] = bb['Middle_BB']
        df['Lower_BB'] = bb['Lower_BB']
        df['STDDEV'] = self._safe_talib_call(talib.STDDEV, ['STDDEV'], df['close'], timeperiod=20)['STDDEV']
        df['MA5'] = self._safe_talib_call(talib.SMA, ['MA5'], df['close'], timeperiod=5)['MA5']
        df['mean_ATR'] = df['ATR'].rolling(20).mean()
        return df