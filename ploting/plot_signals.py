import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_realtime_signals(df, symbol='BTCUSDT', data_range=180, signal_logger=None, output_path=None):
    df = df.copy()

    global_median = df['close'].median()
    df = df[(df['high'] < global_median * 3) & (df['low'] > global_median * 0.3)]

    df_plot = df.iloc[-data_range:].copy()
    if df_plot.index.tz is None:
        raise ValueError("df index must be timezone-aware.")

    # Load signal history
    signal_df = None
    if signal_logger is not None:
        signal_df = signal_logger.get_history()
        if not signal_df.empty:
            signal_df = signal_df.copy()
            signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], utc=True, errors='coerce')
            signal_df.dropna(subset=['timestamp'], inplace=True)
            signal_df['timestamp'] = signal_df['timestamp'].dt.tz_convert(df_plot.index.tz)
            signal_df = signal_df[signal_df['timestamp'] >= df_plot.index.min()]
            signal_df = signal_df.drop_duplicates(subset=['timestamp', 'type'])

    _plot_signals(df_plot, signal_df, symbol, output_path, title_prefix='[Live]')

def plot_backtest_signals(df, symbol='BTCUSDT', signal_df=None, data_range=180, output_path=None):
    df = df.copy()

    global_median = df['close'].median()
    df = df[(df['high'] < global_median * 3) & (df['low'] > global_median * 0.3)]

    pre_context = 5
    df_plot = df.iloc[-(data_range + pre_context):].copy()
    if df_plot.empty:
        print("[Plot] Warning: No data to plot.")
        return

    q1, q3 = df_plot['close'].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_plot = df_plot[(df_plot['low'] >= q1 - 3 * iqr) & (df_plot['high'] <= q3 + 3 * iqr)]

    if df_plot.index.tz is None:
        raise ValueError("df index must be timezone-aware.")

    if signal_df is not None and not signal_df.empty:
        signal_df = signal_df.copy()
        signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], utc=True, errors='coerce')
        signal_df.dropna(subset=['timestamp'], inplace=True)
        signal_df['timestamp'] = signal_df['timestamp'].dt.tz_convert(df_plot.index.tz)
        signal_df = signal_df[
            (signal_df['timestamp'] >= df_plot.index.min()) &
            (signal_df['timestamp'] <= df_plot.index.max())
        ]
        signal_df = signal_df.drop_duplicates(subset=['timestamp', 'type'])

    _plot_signals(df_plot, signal_df, symbol, output_path, title_prefix='[Backtest]')

def _plot_signals(df_plot, signal_df, symbol, output_path, title_prefix=""):
    # Clamp outlier candles using IQR
    q1 = df_plot['close'].quantile(0.25)
    q3 = df_plot['close'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    df_plot = df_plot[(df_plot['low'] >= lower_bound) & (df_plot['high'] <= upper_bound)]

    if df_plot.empty:
        print("[Plot] All candles removed by IQR filter.")
        return

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=['Candlestick + MA', 'RSI', 'ATR', 'Volume'],
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'],
        name='Candlestick',
        increasing_line_color='green', decreasing_line_color='red'
    ), row=1, col=1)

    # MA overlays
    if 'MA5' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA5'], name='MA5', line=dict(color='blue')), row=1, col=1)
    if 'MA20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name='MA20', line=dict(color='purple')), row=1, col=1)

    # RSI
    if 'RSI' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='blue')), row=2, col=1)
        fig.add_hline(y=50, line_dash='dash', line_color='black', row=2, col=1)

    # ATR
    if 'ATR' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ATR'], name='ATR', line=dict(color='orange')), row=3, col=1)
        if 'mean_ATR' in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['mean_ATR'] * 1.2,
                                     name='1.2 * mean_ATR', line=dict(color='red', dash='dash')), row=3, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Volume', marker_color='blue'), row=4, col=1)
    if 'Volume_MA20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Volume_MA20'] * 1.3,
                                 name='1.3 * Volume_MA20', line=dict(color='red', dash='dash')), row=4, col=1)

    # Signal Markers
    if signal_df is not None and not signal_df.empty:
        signal_map = {
            'xgboost_bullish': ('triangle-up', 'green', 1.005),
            'xgboost_bearish': ('triangle-down', 'red', 0.995)
        }
        for _, row in signal_df.iterrows():
            sig_type = row['type']
            if sig_type in signal_map and row['timestamp'] in df_plot.index:
                shape, color, y_adj = signal_map[sig_type]
                label = sig_type.replace('_', ' ').capitalize()
                trigger = row.get('trigger', '')
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['price'] * y_adj],
                    mode='markers',
                    marker=dict(symbol=shape, color=color, size=12),
                    name='',
                    text=[f"{label}: {trigger}"] if trigger else [label],
                    hoverinfo='text+x+y',
                    showlegend=False
                ), row=1, col=1)

    fig.update_layout(
        title=f"{title_prefix} Real-Time 1 Min Signals for {symbol}",
        height=800,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    if output_path:
        fig.write_html(output_path, auto_open=False)
        with open(output_path, "r+", encoding="utf-8") as f:
            content = f.read()
            f.seek(0)
            if "<head>" in content:
                content = content.replace("<head>", "<head>\n<meta http-equiv=\"refresh\" content=\"60\">")
            f.write(content)
            f.truncate()
    else:
        fig.show()
