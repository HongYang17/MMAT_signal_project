import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from tqdm import tqdm

from config import config

# === Constants ===
BINANCE_URL = "https://api.binance.com/api/v3/klines"
LIMIT_PER_CALL = 1000
DEFAULT_START_DATE = datetime(2024, 5, 1)


# === Format raw OHLCV data ===
def format_ohlcv(raw_data) -> pd.DataFrame:
    try:
        df = pd.DataFrame(raw_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception as e:
        print(f"[ERROR] Failed to format OHLCV: {e}")
        return pd.DataFrame()


# === Fetch from Binance with optional progress ===
def fetch_ohlcv(symbol: str, start_time: int, end_time: int, show_progress=False) -> pd.DataFrame:
    all_data = []
    current_ts = start_time
    end_ts = end_time

    total_loops = (end_ts - current_ts) // (60_000 * LIMIT_PER_CALL) + 1
    pbar = tqdm(total=total_loops) if show_progress else None

    while current_ts < end_ts:
        try:
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": current_ts,
                "endTime": end_ts,
                "limit": LIMIT_PER_CALL
            }
            resp = requests.get(BINANCE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            current_ts = data[-1][0] + 60_000

            if pbar:
                pbar.update(1)
            else:
                time.sleep(0.05)  # minimal delay for polite access
        except Exception as e:
            print(f"[ERROR] Binance fetch failed: {e}")
            time.sleep(1)
            continue

    if pbar:
        pbar.close()

    return format_ohlcv(all_data)

# === Update CSV from Binance ===
def update_binance_csv(symbol="BTCUSDT", file_path=None, max_days=395):
    if file_path is None:
        file_path = config.DATA_PATH

    try:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        end_time = now

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
            existing_df.index = existing_df.index.tz_localize("UTC") if existing_df.index.tz is None else existing_df.index
            last_ts = existing_df.index[-1]
            start_time = last_ts + timedelta(minutes=1)
        else:
            existing_df = pd.DataFrame()
            start_time = now - timedelta(days=max_days)

        # Prevent future-dated queries
        if start_time > end_time:
            start_time = end_time - timedelta(minutes=2)

        minutes_to_fetch = int((end_time - start_time).total_seconds() / 60)
        show_progress = minutes_to_fetch > 2000

        new_df = fetch_ohlcv(
            symbol=symbol,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
            show_progress=show_progress
        )

        if new_df.empty:
            print("[Update] No new data returned.")
            return

        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        combined_df.sort_index(inplace=True)

        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=max_days)
        pruned_df = combined_df[combined_df.index >= cutoff]

        pruned_df.to_csv(file_path)

    except Exception as e:
        print(f"[ERROR] update_binance_csv failed: {e}")
        import traceback
        traceback.print_exc()