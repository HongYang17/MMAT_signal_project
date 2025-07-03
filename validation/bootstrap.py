# === validation/bootstrap.py ===

import numpy as np

def bootstrap_accuracy_pvalue(correct_seq, baseline=0.5, B=1000, block_len=20, seed=42):
    """
    Block Bootstrap Test: Evaluate whether prediction accuracy is significantly better than a baseline.

    Parameters:
        correct_seq: list or array of 0s and 1s, representing whether each prediction was correct (1) or not (0)
        baseline: float, the assumed baseline accuracy (e.g., 0.5 for random guessing)
        B: int, number of bootstrap resamples
        block_len: int, block length to preserve temporal dependence in resampling
        seed: int, random seed for reproducibility

    Returns:
        p_value: float, proportion of bootstrap samples where accuracy <= resampled mean accuracy
        acc: float, actual accuracy from the original sequence
        mean_bootstrap: float, average accuracy across bootstrap samples
    """
    np.random.seed(seed)
    correct_seq = np.array(correct_seq)
    n = len(correct_seq)
    indices = np.arange(n)
    boot_means = []

    for _ in range(B):
        sample_idx = []
        nb = int(np.ceil(n / block_len))  # number of blocks to sample
        for _ in range(nb):
            start = np.random.randint(0, n - block_len + 1)
            sample_idx.extend(indices[start: start + block_len])
        sample_idx = sample_idx[:n]  # truncate to exact length
        boot_means.append(correct_seq[sample_idx].mean())

    acc = correct_seq.mean()
    p_value = np.mean([acc <= m for m in boot_means])
    return p_value, acc, np.mean(boot_means)
