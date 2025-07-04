import os

import pandas as pd

from config import config


class SignalHistoryLogger:
    def __init__(self, path=config.SIGNAL_HISTORY_PATH):
        """
        Initialize the logger.
        If path exists, load existing signal log.
        """
        self.path = path
        if path and os.path.exists(path):
            self.df = pd.read_csv(path, parse_dates=['timestamp'])
        else:
            # Create an empty DataFrame with appropriate columns
            self.df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
            self.df = self.df.astype({
                'type': 'str',
                'price': 'float',
                'trigger': 'str'
            })

    def add_signal(self, signal_type, timestamp, price, trigger=None):
        """Add a signal to the log, replacing any existing signal of the same type at the same timestamp."""
        # Remove any existing signal of the same type at the same timestamp
        self.remove_by_type_and_timestamp(signal_type, timestamp)

        # Create a new row as a dictionary
        new_row = {
            'timestamp': pd.to_datetime(timestamp, utc=True),
            'type': signal_type,
            'price': price,
            'trigger': trigger
        }

        # Append the new row to the dataframe
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def remove_by_timestamp(self, timestamp):
        """Remove rows with the given timestamp."""
        self.df = self.df[self.df['timestamp'] != timestamp]

    def remove_by_type_and_timestamp(self, signal_type, timestamp):
        """Remove signals of a particular type and timestamp."""
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

    def has_signal(self, timestamp, signal_type):
        """Check if a signal of the given type exists at the given timestamp."""
        if self.df.empty:
            return False
        return not self.df[
            (self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type)
            ].empty

    def get_history(self):
        """Return a copy of the signal history DataFrame."""
        return self.df.copy()

    def save_to_csv(self, path=None):
        """Save the signal history to a CSV file."""
        path = path or self.path

        # Check if the directory exists, if not, create it
        if path and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Only save if the DataFrame is not empty
        if not self.df.empty:
            self.df.to_csv(path, index=False)
            print(f"[Logger] Saved {len(self.df)} signals to {path}")
        else:
            print("[Logger] No signals to save.")

    def remove_opposite_signal(self, timestamp, signal_type):
        """Remove the opposite signal of the given type at the timestamp."""
        opposite = {
            'xgboost_bullish': 'xgboost_bearish',
            'xgboost_bearish': 'xgboost_bullish'
        }
        if signal_type in opposite:
            self.remove_by_type_and_timestamp(opposite[signal_type], timestamp)