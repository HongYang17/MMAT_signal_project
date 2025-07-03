import pandas as pd
import os

class SignalHistoryLogger:
    def __init__(self, filename='signal_history.csv'):
        self.filename = filename
        if os.path.exists(self.filename):
            self.df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        else:
            self.df = pd.DataFrame(columns=['timestamp', 'type', 'price', 'trigger'])

    def add_signal(self, signal_type, timestamp, price, trigger=None):
        """
        Add a new signal entry.
        If the same signal type already exists at the same timestamp, remove it first to avoid duplication.
        """
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

        new_row = {
            'timestamp': timestamp,
            'type': signal_type,
            'price': price,
            'trigger': trigger
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def remove_by_timestamp(self, timestamp):
        """
        Remove all signals at the specified timestamp, regardless of type.
        """
        self.df = self.df[self.df['timestamp'] != timestamp]

    def has_signal(self, timestamp, signal_type):
        """
        Check whether a specific type of signal already exists at the given timestamp.
        """
        if self.df.empty:
            return False
        return not self.df[
            (self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type)
        ].empty

    def get_history(self):
        """
        Return a copy of the signal history DataFrame.
        """
        return self.df.copy()

    def save_to_csv(self, filename=None):
        """
        Save the current signal history to a CSV file.
        """
        path = filename if filename else self.filename
        self.df.to_csv(path, index=False)

    def remove_by_type_and_timestamp(self, signal_type, timestamp):
        """
        Remove a specific signal by type and timestamp.
        """
        self.df = self.df[~((self.df['timestamp'] == timestamp) & (self.df['type'] == signal_type))]

    def remove_opposite_signal(self, timestamp, signal_type):
        """
        Automatically remove the opposite signal type at the same timestamp.
        For example, if the current signal is bullish, remove the bearish one.
        """
        opposite = {
            'xgboost_bullish': 'xgboost_bearish',
            'xgboost_bearish': 'xgboost_bullish'
        }
        if signal_type in opposite:
            self.remove_by_type_and_timestamp(opposite[signal_type], timestamp)
