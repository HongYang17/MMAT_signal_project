import atexit
import os
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from binance.client import Client
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# === Define Base Directory ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# === Safe Imports ===
from config.load_env import load_keys
from signals.prediction_logger import PredictionLogger
from signals.signal_logger import SignalHistoryLogger

# === Load XGBoost model and selected features ===
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'improved_signal_model.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'selected_features.pkl')

try:
    xgb_model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURE_NAMES_PATH)
    print(f"Loaded XGBoost model with {len(selected_features)} features")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgb_model = None
    selected_features = []

# === Load API Keys ===
try:
    keys = load_keys()
except Exception:
    # Fallback if config.load_env is missing or fails
    dotenv_path = os.path.join(BASE_DIR, '.env')
    load_dotenv(dotenv_path)
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')
    if not api_key or not api_secret:
        raise ValueError("No API keys found. Set BINANCE_API_KEY and BINANCE_SECRET_KEY in your .env file.")
    keys = {'api_key': api_key, 'secret_key': api_secret}

# === Initialize Binance Client ===
client = Client(keys['api_key'], keys['secret_key'])

# === Load historical CSV data ===
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        print(f"Total K-lines loaded: {len(df)}")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return None

# === Fetch real-time Binance data ===
def fetch_binance_data(api_key, api_secret, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000):
    try:
        client = Client(api_key, api_secret)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Singapore')
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        print(f"Fetched {len(df)} K-lines from Binance API")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return None

def calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate core technical indicators and add them to the DataFrame.

    Parameters:
        df (pd.DataFrame): Must contain columns ['open', 'high', 'low', 'close', 'volume']

    Returns:
        pd.DataFrame: DataFrame with new indicator columns added
    """
    # === Momentum & Oscillator ===
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)

    # === Trend-Following Indicators ===
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
    df['EMA200'] = talib.EMA(df['close'], timeperiod=200)
    df['SMA20'] = talib.SMA(df['close'], timeperiod=20)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

    # === Volume-Based Indicators ===
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['Volume_MA20'] = talib.SMA(df['volume'], timeperiod=20)

    # === Volatility Indicators ===
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)
    df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=20)

    # === Moving Averages (fast + volatility smoothing) ===
    df['MA5'] = talib.SMA(df['close'], timeperiod=5)
    df['mean_ATR'] = df['ATR'].rolling(window=20, min_periods=10).mean()

    # === Drop rows with NA values from indicator initialization ===
    df.dropna(inplace=True)

    return df

import talib
import pandas as pd

def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect candlestick patterns using TA-Lib, store raw outputs (±100),
    and derive standardized signal columns (-1, 0, +1).

    Returns:
        df (pd.DataFrame): DataFrame with pattern signals and aggregated candle signal metrics.
    """
    # === Define candlestick patterns ===
    patterns = {
        # Bullish
        'Hammer': talib.CDLHAMMER,
        'InvertedHammer': talib.CDLINVERTEDHAMMER,
        'BullishEngulfing': talib.CDLENGULFING,
        'PiercingLine': talib.CDLPIERCING,
        'MorningStar': talib.CDLMORNINGSTAR,
        'DragonflyDoji': talib.CDLDRAGONFLYDOJI,
        'LongLine': talib.CDLLONGLINE,
        'ThreeLineStrike': talib.CDL3LINESTRIKE,

        # Bearish
        'HangingMan': talib.CDLHANGINGMAN,
        'ShootingStar': talib.CDLSHOOTINGSTAR,
        'BearishEngulfing': talib.CDLENGULFING,
        'DarkCloudCover': talib.CDLDARKCLOUDCOVER,
        'EveningDojiStar': talib.CDLEVENINGDOJISTAR,
        'EveningStar': talib.CDLEVENINGSTAR,
        'GravestoneDoji': talib.CDLGRAVESTONEDOJI,
    }

    # === Apply raw TA-Lib pattern functions (±100 or 0) ===
    for name, func in patterns.items():
        df[name] = func(df['open'], df['high'], df['low'], df['close'])

    # === Classify into bullish / bearish for signal translation ===
    bullish = {'Hammer', 'InvertedHammer', 'BullishEngulfing', 'PiercingLine',
               'MorningStar', 'DragonflyDoji', 'LongLine', 'ThreeLineStrike'}
    bearish = {'HangingMan', 'ShootingStar', 'BearishEngulfing', 'DarkCloudCover',
               'EveningDojiStar', 'EveningStar', 'GravestoneDoji'}

    # === Translate to -1 / 0 / +1 signals ===
    for name in patterns:
        raw_col = df[name]
        if name in bullish:
            df[f'Signal_{name}'] = raw_col.apply(lambda x: 1 if x == 100 else 0)
        elif name in bearish:
            df[f'Signal_{name}'] = raw_col.apply(lambda x: -1 if x == -100 else 0)
        else:
            # Fallback in case of mixed patterns
            df[f'Signal_{name}'] = raw_col.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # === Aggregate signal counts ===
    signal_cols = [f'Signal_{name}' for name in patterns]
    df['total_bullish_signals'] = df[signal_cols].sum(axis=1).clip(lower=0)
    df['total_bearish_signals'] = (-df[signal_cols]).clip(lower=0).sum(axis=1)
    df['net_candle_signal'] = df['total_bullish_signals'] - df['total_bearish_signals']

    return df


import pandas as pd


def calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate additional engineered features from price, volume, and indicator data.

    Adds ratio, lag, percentage change, and interaction features for ML modeling.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    features = {}

    # === Ratio-Based Features ===
    features['close_to_high'] = (df['high'] - df['close']) / df['high'].replace(0, pd.NA)
    features['close_to_low'] = (df['close'] - df['low']) / df['close'].replace(0, pd.NA)
    features['price_range'] = (df['high'] - df['low']) / df['close'].replace(0, pd.NA)
    features['volatility_ratio'] = df['ATR'] / df['close'].rolling(20).mean().shift(1)

    # === Price & Volume Movement Features ===
    features['price_change'] = df['close'].pct_change()
    features['volume_change'] = df['volume'].pct_change()
    features['volume_ratio'] = df['volume'] / df['Volume_MA'].replace(0, pd.NA)

    # === Indicator Divergence Features ===
    features['rsi_divergence'] = df['RSI'] - df['RSI'].rolling(5).mean().shift(1)
    features['macd_hist'] = df['MACD'] - df['MACD_signal']
    features['distance_to_upper_bb'] = (df['Upper_BB'] - df['close']) / df['close'].replace(0, pd.NA)
    features['distance_to_lower_bb'] = (df['close'] - df['Lower_BB']) / df['close'].replace(0, pd.NA)
    features['trend_power'] = df['ADX'] * (df['PLUS_DI'] - df['MINUS_DI'])

    # === Lag Features ===
    lag_cols = ['close', 'volume', 'RSI', 'MACD', 'ATR', 'ADX']
    for col in lag_cols:
        for lag in [1, 2, 3, 5, 10]:
            features[f'{col}_lag{lag}'] = df[col].shift(lag)

    # === Percent Change Features ===
    pct_cols = ['RSI', 'MACD', 'ATR', 'volume', 'close']
    for col in pct_cols:
        features[f'{col}_pct_change'] = df[col].pct_change()

    # === Cross & Binary Interaction Features ===
    features['macd_histogram'] = df['MACD'] - df['MACD_signal']  # same as 'macd_hist'
    features['di_crossover'] = (df['PLUS_DI'] > df['MINUS_DI']).astype(int)

    # === Combine ===
    df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    # Ensure aggregation columns exist (important for downstream signal weighting)
    for col in ['net_candle_signal', 'total_bullish_signals', 'total_bearish_signals']:
        if col not in df.columns:
            df[col] = 0

    # Drop rows with any NA caused by rolling/lags
    df.dropna(inplace=True)

    return df

def generate_xgboost_signals(df, signal_logger=None):
    """
    Generate trading signals using a pre-trained XGBoost model.
    This version includes safety checks and optional signal logging.

    Parameters:
        df (pd.DataFrame): Input data with raw features
        signal_logger (SignalHistoryLogger, optional): Logger to record signals

    Returns:
        pd.DataFrame: DataFrame with added signal columns
    """
    if xgb_model is None:
        print("XGBoost model not loaded. Skipping signal generation.")
        return df

    # Initialize signal columns
    df['xgboost_signal'] = 0
    df['xgboost_direction'] = 'NONE'
    df['xgboost_confidence'] = 0.0

    # Compute necessary features
    df = calculate_basic_indicators(df)
    df = calculate_patterns(df)
    df = calculate_additional_features(df)

    # Check for missing features
    missing_features = set(selected_features) - set(df.columns)
    if missing_features:
        print(f"Missing features for XGBoost model: {missing_features}")
        return df

    # Use the second-to-last row (last completed candle)
    i = len(df) - 2
    if i < 0 or i >= len(df):
        print("Not enough valid rows for prediction after feature engineering.")
        return df

    try:
        # Prepare features for prediction
        features_df = df.loc[[df.index[i]], selected_features].copy()

        # Check for NaN
        if features_df.isnull().any().any():
            nan_cols = features_df.columns[features_df.isnull().any()].tolist()
            print(f"Skipping prediction: NaN found in features at index {df.index[i]} in columns: {nan_cols}")
            return df

        # Predict
        prediction = xgb_model.predict(features_df)[0]
        proba = xgb_model.predict_proba(features_df)[0][1]  # probability of class 1 (UP)

        print(f"XGBoost prediction at {df.index[i]}: {'UP' if prediction == 1 else 'DOWN'} (Confidence: {proba:.2%})")

        # Update DataFrame
        if prediction == 1:
            df.loc[df.index[i], 'xgboost_signal'] = 1
            df.loc[df.index[i], 'xgboost_direction'] = 'UP'
            df.loc[df.index[i], 'xgboost_confidence'] = proba

            if signal_logger:
                signal_logger.add_signal(
                    'xgboost_bullish',
                    df.index[i],
                    df['close'].iloc[i],
                    trigger=f"Confidence: {proba:.2%}"
                )

        elif prediction == 0:
            df.loc[df.index[i], 'xgboost_signal'] = -1
            df.loc[df.index[i], 'xgboost_direction'] = 'DOWN'
            df.loc[df.index[i], 'xgboost_confidence'] = 1 - proba

            if signal_logger:
                signal_logger.add_signal(
                    'xgboost_bearish',
                    df.index[i],
                    df['close'].iloc[i],
                    trigger=f"Confidence: {(1 - proba):.2%}"
                )

        else:
            print("Warning: XGBoost output was neither 0 nor 1 — treated as NEUTRAL.")

    except Exception as e:
        print(f"Error in XGBoost prediction: {e}")
        import traceback
        traceback.print_exc()

    return df

import pandas as pd

def evaluate_patterns(df: pd.DataFrame, patterns_dict: dict, window: int = 5, threshold: float = 0.001) -> dict:
    """
    Evaluate the predictive accuracy of candlestick pattern signals.

    For each pattern signal, computes the forward return after `window` bars
    and checks if the movement aligns with the signal direction based on the `threshold`.

    Parameters:
        df (pd.DataFrame): Must include pattern signals (e.g., Signal_Hammer) and 'close' prices
        patterns_dict (dict): Keys = pattern names, used to locate signal columns (e.g., Signal_Hammer)
        window (int): Number of bars to look ahead for return calculation
        threshold (float): Minimum return (absolute) to count as a correct prediction

    Returns:
        dict: Mapping of pattern name → {'accuracy', 'total_signals', 'correct_signals'}
    """
    results = {}

    # Forward return calculation
    df = df.copy()
    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    for name in patterns_dict.keys():
        signal_col = f'Signal_{name}'
        if signal_col not in df.columns:
            continue

        signals = df[df[signal_col] != 0]
        total = len(signals)

        if total == 0:
            results[name] = {'accuracy': 0.0, 'total_signals': 0, 'correct_signals': 0}
            continue

        # Match signal direction with return
        correct = signals[
            ((signals[signal_col] == 1) & (signals['return'] >= threshold)) |
            ((signals[signal_col] == -1) & (signals['return'] <= -threshold))
        ]

        accuracy = len(correct) / total * 100

        results[name] = {
            'accuracy': round(accuracy, 2),
            'total_signals': total,
            'correct_signals': len(correct)
        }

    return results

def plot_realtime_signals(df, symbol='BTCUSDT', data_range=50, output_dir=None, signal_logger=None):
    """
    Plot real-time candlestick chart with indicators and overlayed trading signals.
    Saves to validation/realtime_signals_<symbol>.html with auto-refresh enabled.
    """

    if signal_logger is None:
        raise ValueError("signal_logger must be provided")

    # Default to saving in validation/plots folder
    if output_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(base_dir, 'validation', 'plots')

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    df_plot = df.iloc[-data_range:].copy()

    # Latest forming candle hover text
    df_plot['hover_text'] = np.where(
        df_plot.index == df.index[-1],
        ' Latest forming candle (not evaluated)',
        ''
    )

    print(f"Plotting real-time chart for last {data_range} candles")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Candlestick + MA', 'RSI', 'ATR', 'Volume'],
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # === Candlestick + MAs ===
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Candlestick',
        increasing=dict(line=dict(color='green')),
        decreasing=dict(line=dict(color='red'))
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['MA5'], mode='lines', name='5 MA', line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['MA20'], mode='lines', name='20 MA', line=dict(color='purple')
    ), row=1, col=1)

    # === Forming candle marker ===
    forming_candle = df_plot[df_plot.index == df.index[-1]]
    if not forming_candle.empty:
        fig.add_trace(go.Scatter(
            x=forming_candle.index,
            y=forming_candle['close'] * 1.002,
            mode='markers',
            name='Forming Candle',
            marker=dict(symbol='circle', color='gray', size=8, opacity=0.3),
            text=forming_candle['hover_text'],
            hoverinfo='text+x+y',
            hoverlabel=dict(bgcolor='lightgray'),
            showlegend=False
        ), row=1, col=1)

    # === Historical signals from logger ===
    signal_df = signal_logger.get_history()
    print(f"Signal history loaded: {len(signal_df)} records")

    signal_map = {
        'xgboost_bullish': ('triangle-up', 'green', 1.005),
        'xgboost_bearish': ('triangle-down', 'red', 0.995),
    }

    if not signal_df.empty:
        try:
            signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], utc=True)
            signal_df['timestamp'] = signal_df['timestamp'].dt.tz_convert(df_plot.index.tz)
        except Exception as e:
            print(f"Error converting signal timestamps: {e}")
            signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])

        signal_df = signal_df[signal_df['timestamp'] >= df_plot.index[0]]
        signal_df = signal_df.drop_duplicates(subset=['timestamp', 'type'])

        for _, row in signal_df.iterrows():
            sig_type = row['type']
            if sig_type in signal_map and row['timestamp'] in df_plot.index:
                symbol_shape, color, y_factor = signal_map[sig_type]
                trigger_text = (
                    f"{sig_type.replace('_', ' ').capitalize()}: {row['trigger']}"
                    if 'trigger' in row and pd.notna(row['trigger']) else sig_type.replace('_', ' ').capitalize()
                )
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['price'] * y_factor],
                    mode='markers',
                    marker=dict(symbol=symbol_shape, color=color, size=12),
                    name='',
                    text=[trigger_text],
                    hoverinfo='text+x+y',
                    showlegend=False
                ), row=1, col=1)

    # === RSI ===
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='blue')
    ), row=2, col=1)

    fig.add_hline(y=50, line_dash='dash', line_color='black', row=2, col=1)

    # === ATR ===
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['ATR'],
        mode='lines',
        name='ATR',
        line=dict(color='orange')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['mean_ATR'] * 1.2,
        mode='lines',
        name='1.2 * mean_ATR',
        line=dict(color='red', dash='dash')
    ), row=3, col=1)

    # === Volume ===
    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['volume'],
        name='Volume',
        marker=dict(color='blue')
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot['Volume_MA20'] * 1.3,
        mode='lines',
        name='1.3 * Volume_MA20',
        line=dict(color='red', dash='dash')
    ), row=4, col=1)

    # === Final layout ===
    fig.update_layout(
        title=f'Real-Time 5 Min Signals for {symbol}',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=800,
        template='plotly_white'
    )

    # === Save to HTML with auto-refresh ===
    html_path = os.path.join(output_dir, f'realtime_signals_{symbol}.html')
    try:
        html_content = fig.to_html(include_plotlyjs='cdn')
        html_content = html_content.replace(
            '<head>',
            '<head><meta http-equiv="refresh" content="300">'
        )
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"Plot saved to: {html_path}")
    except Exception as e:
        print(f"Error saving HTML plot: {e}")

# === Global loggers ===
logger = PredictionLogger()
signal_logger = SignalHistoryLogger(filename=os.path.join('validation', 'signal_history.csv'))

# === Save logs at program exit ===
atexit.register(lambda: logger.save_to_csv(os.path.join('validation', 'TestLive_prediction_log.csv')))
atexit.register(lambda: signal_logger.save_to_csv(os.path.join('validation', 'signal_history.csv')))

def update_signal(signal_logger, signal_type, timestamp, price, confidence_str):
    """Remove opposite signal and log new one."""
    signal_logger.remove_opposite_signal(timestamp, signal_type)
    if not signal_logger.has_signal(timestamp, signal_type):
        signal_logger.add_signal(signal_type, timestamp, price, trigger=confidence_str)

def run_realtime_signals(api_key, api_secret, symbol='BTCUSDT',
                         interval=Client.KLINE_INTERVAL_1MINUTE,
                         limit=1000, sleep_seconds=300,
                         signal_logger=None, prediction_logger=None, debug=False):
    """
    Fetch live Binance data, generate XGBoost signals, log predictions, and visualize.
    """

    print("Starting real-time signal generation. Press Ctrl+C to stop.")

    while True:
        try:
            df = fetch_binance_data(api_key, api_secret, symbol, interval, limit)
            if df is None or df.empty:
                print("Failed to fetch data. Retrying in 60 seconds.")
                time.sleep(60)
                continue

            df = generate_xgboost_signals(df, signal_logger=signal_logger)

            latest_ts = df.index[-2]
            close_now = df['close'].iloc[-2]
            xgboost_signal = df['xgboost_signal'].iloc[-2]
            xgboost_confidence = df['xgboost_confidence'].iloc[-2]

            prediction = "NEUTRAL"
            if xgboost_signal == 1:
                prediction = "UP"
                print(f"\nXGBoost Signal: BULLISH (Confidence: {xgboost_confidence:.2%})")
                update_signal(signal_logger, 'xgboost_bullish', latest_ts, close_now,
                              confidence_str=f"Confidence: {xgboost_confidence:.2%}")

            elif xgboost_signal == -1:
                prediction = "DOWN"
                print(f"\nXGBoost Signal: BEARISH (Confidence: {(1 - xgboost_confidence):.2%})")
                update_signal(signal_logger, 'xgboost_bearish', latest_ts, close_now,
                              confidence_str=f"Confidence: {(1 - xgboost_confidence):.2%}")
            else:
                print("\nXGBoost Signal: NEUTRAL")

            print(f"Prediction for next 5min: {prediction}")

            if len(df) >= 2:
                ts = df.index[-1]
                close_future = df['close'].iloc[-1]
                logger.record_prediction(ts, prediction, close_future, close_now)
                print(f"Current Hit Rate: {logger.get_hit_rate():.2%}")

            plot_realtime_signals(df, symbol=symbol, data_range=50, signal_logger=signal_logger)

            print(f"Waiting {sleep_seconds} seconds for next update...\n")
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("\nStopped real-time signal generation. Exiting...")
            break
        except Exception as e:
            print(f"Error in real-time loop: {e}. Retrying in 60 seconds.")
            time.sleep(60)

def main(realtime=True, debug=False):
    """
    Main function to run candlestick pattern analysis and signal generation.
    Supports both real-time data from Binance API and historical data from CSV.
    """

    # Configuration
    symbol = 'BTCUSDT'
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'data', 'btc_1min.csv')
    signal_log_path = os.path.join(base_dir, 'validation', 'signal_history.csv')
    prediction_log_path = os.path.join(base_dir, 'validation', 'TestLive_prediction_log.csv')

    # Initialize loggers
    signal_logger = SignalHistoryLogger(filename=signal_log_path)
    prediction_logger = PredictionLogger()

    # Save logs on program exit
    atexit.register(lambda: prediction_logger.save_to_csv(prediction_log_path))
    atexit.register(lambda: signal_logger.save_to_csv(signal_log_path))

    if realtime:
        try:
            keys = load_keys()
            api_key = keys['api_key']
            api_secret = keys['secret_key']
        except Exception as e:
            print(f"Failed to load API keys: {e}")
            return

        # Run real-time live loop
        run_realtime_signals(
            api_key, api_secret,
            symbol=symbol,
            signal_logger=signal_logger,
            prediction_logger=prediction_logger,
            debug=debug
        )

    else:
        try:
            df = load_data(csv_path)
            if df is None or df.empty:
                print("Failed to load CSV data.")
                return
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return

        # Feature engineering + signal generation
        df = calculate_basic_indicators(df)
        df = calculate_patterns(df)
        df = calculate_additional_features(df)
        df = generate_xgboost_signals(df, signal_logger=signal_logger)

        # Evaluate signals
        accuracy_results = evaluate_patterns(df)
        print("\n--- Pattern Signal Accuracy ---")
        for name, metrics in sorted(accuracy_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{name} - Accuracy: {metrics['accuracy']:.2f}%, "
                  f"Total: {metrics['total_signals']}, "
                  f"Correct: {metrics['correct_signals']}")

        # Plot result
        plot_realtime_signals(df, symbol=symbol, data_range=50, signal_logger=signal_logger)

if __name__ == "__main__":
    main(realtime=True, debug=False)
