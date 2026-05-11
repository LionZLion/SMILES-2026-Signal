import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers, shifted_window, MODEL_SUBSET

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def my_canceller(rx, tx_n, fs):
    
    def rank1_from_band_matrix(band_matrix):
        # from task_and_baseline
        cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
        _, vecs = np.linalg.eigh(cov)
        shared = band_matrix @ vecs[:, -1]
        denom = np.vdot(shared, shared) + 1e-30
        return np.column_stack([
            (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
            for ch in range(band_matrix.shape[1])
        ])
    
    n = len(rx)
    helpers = build_task_helpers(tx_n, fs, n)

    def fit_tx(signal, candidates, LAGS, alpha=1e-2):
        START, STOP = MODEL_SUBSET.start, MODEL_SUBSET.stop
        n_feat = len(candidates) * len(LAGS)
        n_fit = STOP - START

        X_fit = np.empty((n_fit, n_feat), dtype=np.complex128)
        for ci, c in enumerate(candidates):
            for li, lag in enumerate(LAGS):
                X_fit[:, ci * len(LAGS) + li] = shifted_window(c, lag, START, STOP)

        gram = X_fit.conj().T @ X_fit + alpha * np.eye(n_feat)
        XH = X_fit.conj().T
        coefs = np.zeros((n_feat, 4), dtype=np.complex128)
        for ch in range(4):
            y = helpers["score_filter"](signal[:, ch])[START:STOP]
            coefs[:, ch] = np.linalg.solve(gram, XH @ y)
        del X_fit, XH

        pred = np.zeros((n, 4), dtype=np.complex128)
        for start in range(0, n, 20_000):
            stop = min(start + 20_000, n)
            X_batch = np.empty((stop - start, n_feat), dtype=np.complex128)
            for ci, c in enumerate(candidates):
                for li, lag in enumerate(LAGS):
                    X_batch[:, ci * len(LAGS) + li] = shifted_window(c, lag, start, stop)
            pred[start:stop] = X_batch @ coefs
        return pred

    def fit_rank1(signal):
        """Rank-1 from task_and_baseline"""
        band = np.column_stack([
            helpers["score_filter"](signal[:, ch]) for ch in range(4)
        ])
        return rank1_from_band_matrix(band)

    # New Features
    candidates = []
    for i in range(6):
        for j in range(6):
            candidates.append(helpers["score_filter"](tx_n[:,i]**2 * tx_n[:,j].conj()))
            candidates.append(helpers["score_filter"](tx_n[:,i] * np.abs(tx_n[:,j])**2))
    LAGS = list(range(-10, 11))

    print("=== Iteration 1 ===")
    tx_pred1 = fit_tx(rx, candidates, LAGS)
    residual1 = rx - tx_pred1
    rank1_1 = fit_rank1(residual1)
    residual1 = residual1 - rank1_1

    print("=== Iteration 2 ===")
    tx_pred2 = fit_tx(residual1, candidates, LAGS)
    # residual2 = residual1 - tx_pred2
    print("=== End of fit ===")
    return rx - tx_pred1 - rank1_1 - tx_pred2


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, my_canceller(rx, tx_n, Fs), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
