# === realtime/plot_signals.py ===

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_realtime_signals(df, symbol='BTCUSDT', signal_df=None, data_range=50, output_path=None):
    """
    Plot real-time candlestick chart with overlayed signal markers.

    Parameters:
        df : pd.DataFrame with OHLCV + indicators (index=timestamp)
        symbol : str, trading symbol
        signal_df : pd.DataFrame with ['timestamp', 'type', 'price', 'trigger']
        data_range : int, number of bars to show
        output_path : str, optional HTML file path
    """
    df_plot = df.iloc[-data_range:].copy()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        subplot_titles=("Candlestick", "RSI", "ATR", "Volume")
    )

    # === Candlestick ===
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'],
        name='Candles', increasing_line_color='green', decreasing_line_color='red'
    ), row=1, col=1)

    if 'MA5' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA5'],
                                 name='MA5', line=dict(color='blue')), row=1, col=1)

    if 'MA20' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'],
                                 name='MA20', line=dict(color='purple')), row=1, col=1)

    # === RSI ===
    if 'RSI' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='blue')), row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", row=2, col=1)

    # === ATR ===
    if 'ATR' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ATR'], name='ATR', line=dict(color='orange')), row=3, col=1)

    # === Volume ===
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], name='Volume', marker_color='gray'), row=4, col=1)

    # === Signal overlay ===
    if signal_df is not None:
        signal_df = signal_df.copy()
        signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])
        signal_df = signal_df[signal_df['timestamp'] >= df_plot.index[0]]

        for _, row in signal_df.iterrows():
            if row['type'] == 'xgboost_bullish':
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']], y=[row['price'] * 1.005],
                    mode='markers', name='BUY', marker=dict(symbol='triangle-up', color='green', size=12),
                    text=row.get('trigger', ''), hoverinfo='text+x+y', showlegend=False
                ), row=1, col=1)
            elif row['type'] == 'xgboost_bearish':
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']], y=[row['price'] * 0.995],
                    mode='markers', name='SELL', marker=dict(symbol='triangle-down', color='red', size=12),
                    text=row.get('trigger', ''), hoverinfo='text+x+y', showlegend=False
                ), row=1, col=1)

    fig.update_layout(
        title=f"{symbol} - Real-Time Signals",
        xaxis_rangeslider_visible=False,
        template='plotly_white', height=800
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        fig.show()
