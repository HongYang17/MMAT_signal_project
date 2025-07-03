# === data/data_fetcher.py ===

import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm


def get_binance_1m_data(symbol="BTCUSDT", start_time=None, end_time=None):
    """
    Fetch 1-minute OHLCV data from Binance REST API for a given time range.
    """
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000

    all_data = []
    current_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    print(f"Fetching {symbol} 1m data from Binance...")
    pbar = tqdm(total=(end_ts - current_ts) // (60 * 1000 * limit) + 1)

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "limit": limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Request failed: {e}")
            break

        if not data:
            break

        all_data.extend(data)
        current_ts = data[-1][0] + 60 * 1000
        time.sleep(0.1)
        pbar.update(1)

    pbar.close()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df


def update_binance_csv(symbol="BTCUSDT", file_path="data/BTCUSDT_1min_2024-05-01_to_now.csv", start_date=datetime(2024, 5, 1)):
    """
    Append new 1-minute Binance data to existing CSV, or download from scratch if no file exists.
    """
    now = datetime.utcnow()

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
        last_timestamp = existing_df.index[-1]
        start_time = last_timestamp + timedelta(minutes=1)
        print(f"Found existing file. Last timestamp: {last_timestamp}. Downloading from there.")
    else:
        existing_df = pd.DataFrame()
        start_time = start_date
        print("No existing file. Downloading from scratch...")

    new_df = get_binance_1m_data(symbol, start_time=start_time, end_time=now)

    if not new_df.empty:
        updated_df = pd.concat([existing_df, new_df])
        updated_df = updated_df[~updated_df.index.duplicated(keep='first')]
        updated_df.sort_index(inplace=True)
        updated_df.to_csv(file_path)
        print(f"Saved updated data to {file_path}. Total rows: {len(updated_df)}")
    else:
        print("No new data downloaded.")


def fetch_binance_data(symbol="BTCUSDT", interval="1m", limit=1000):
    """
    Fetch the latest Binance OHLCV data for live prediction (no start_time required).
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df