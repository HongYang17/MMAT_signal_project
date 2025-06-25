import os
import pandas as pd


class SignalHistoryLogger:
    """
    Tracks historical trading signals with timestamps, type, price, and optional trigger text.
    """

    def __init__(self, filename=None):
        # Default to ../validation/signal_history.csv
        if filename is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            filename = os.path.join(base_dir, 'validation', 'signal_history.csv')

        self.filename = filename

        try:
            if os.path.exists(self.filename):
                self.df = pd.read_csv(self.filename, parse_dates=['timestamp'])
            else:
                self.df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])
        except Exception as e:
            print(f"[SignalLogger] Failed to load existing file: {e}")
            self.df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])

    def add_signal(self, signal_type, timestamp, price, trigger=None):
        """Adds a signal after removing duplicates at same timestamp and type."""
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

        new_row = {
            'timestamp': timestamp,
            'type': signal_type,
            'price': price,
            'trigger': trigger
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def remove_by_timestamp(self, timestamp):
        """Removes all signals at a given timestamp."""
        self.df = self.df[self.df['timestamp'] != timestamp]

    def has_signal(self, timestamp, signal_type):
        """Checks whether a signal exists for a given type and timestamp."""
        if self.df.empty:
            return False
        return not self.df[
            (self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type)
        ].empty

    def get_history(self):
        """Returns a copy of the full signal log."""
        return self.df.copy()

    def save_to_csv(self, filename=None):
        """Saves the signal log to CSV. Creates folders if necessary."""
        path = filename if filename else self.filename
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.df.to_csv(path, index=False)
            print(f"[SignalLogger] Saved {len(self.df)} signals to {path}")
        except Exception as e:
            print(f"[SignalLogger] Failed to save CSV: {e}")

    def remove_by_type_and_timestamp(self, signal_type, timestamp):
        """Removes a signal by specific type and timestamp."""
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

    def remove_opposite_signal(self, timestamp, signal_type):
        """Automatically removes the opposite directional signal if it exists."""
        opposite = {
            'xgboost_bullish': 'xgboost_bearish',
            'xgboost_bearish': 'xgboost_bullish'
        }
        if signal_type in opposite:
            self.remove_by_type_and_timestamp(opposite[signal_type], timestamp)