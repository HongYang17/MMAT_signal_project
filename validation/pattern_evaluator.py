# === validation/pattern_evaluator.py ===

import pandas as pd

def evaluate_patterns(df, patterns_dict, window=5, threshold=0.001):
    """
    Evaluate the accuracy of each candlestick pattern signal.
    
    Parameters:
        df : DataFrame with pattern columns and OHLCV
        patterns_dict : dict of pattern name -> talib function
        window : int, how many bars ahead to check return
        threshold : float, min return to count as a correct prediction
    
    Returns:
        dict of pattern -> {accuracy, total_signals, correct_signals}
    """
    results = {}
    df['next_close'] = df['close'].shift(-window)
    df['return'] = (df['next_close'] - df['close']) / df['close']

    for name in patterns_dict.keys():
        signal_col = f"Signal_{name}"
        if signal_col not in df.columns:
            continue

        signals = df[df[signal_col] != 0]
        total = len(signals)

        if total == 0:
            results[name] = {'accuracy': 0, 'total_signals': 0, 'correct_signals': 0}
            continue

        correct = len(signals[
            ((signals[signal_col] == 1) & (signals['return'] >= threshold)) |
            ((signals[signal_col] == -1) & (signals['return'] <= -threshold))
        ])
        acc = correct / total * 100

        results[name] = {
            'accuracy': acc,
            'total_signals': total,
            'correct_signals': correct
        }

    return results


def summarize_pattern_performance(results_dict):
    """
    Convert raw pattern result dict to a sorted pandas DataFrame
    """
    df = pd.DataFrame(results_dict).T
    df = df.sort_values(by='accuracy', ascending=False)
    return df
