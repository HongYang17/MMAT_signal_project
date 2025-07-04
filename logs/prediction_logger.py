import os

import pandas as pd

from config import config


class PredictionLogger:
    def __init__(self, path=config.PREDICTION_LOG_PATH, autosave=False):
        """
        Initialize the prediction logger.
        If autosave=True, save after each prediction.
        """
        self.log = []
        self.path = path
        self.autosave = autosave

    def record_prediction(self, timestamp, prediction, close_now, close_prev, confidence=None):
        # Only record if prediction is UP or DOWN (you can add more conditions if needed)
        if prediction.startswith("UP") or prediction.startswith("DOWN"):
            try:
                # Ensure that close_now and close_prev are valid and non-zero
                if close_prev == 0 or close_now == 0:
                    print(f"[Logger] Invalid data: close_now or close_prev is zero. Skipping prediction.")
                    return

                ret = (close_now - close_prev) / close_prev
                hit = ((prediction == "UP" and ret > 0) or (prediction == "DOWN" and ret < 0))
                self.log.append({
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'return': ret,
                    'hit': int(hit),
                    'confidence': confidence,
                    'close_now': close_now,
                    'close_prev': close_prev
                })

                # Save immediately if autosave is enabled
                if self.autosave:
                    self.save_to_csv()

            except Exception as e:
                print(f"[Logger] Error recording prediction: {e}")

    def get_hit_rate(self):
        if not self.log:
            return 0.0
        return sum(entry['hit'] for entry in self.log) / len(self.log)

    def to_dataframe(self):
        return pd.DataFrame(self.log)

    def save_to_csv(self, path=None):
        path = path or self.path
        df = self.to_dataframe()

        if df.empty:
            print("[Logger] No predictions to save.")
            return

        # Ensure the path exists or create the directory
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Save the data to CSV
        df.to_csv(path, index=False)
        print(f"[Logger] Saved {len(df)} predictions to {path}")
