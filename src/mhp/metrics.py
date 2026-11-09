import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def reg_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=r2_score(y, p) if len(y) > 1 else np.nan, Bias=float(np.mean(p - y)), n=len(y))


def bootstrap_ci(fn, n, B=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of fn(idx) over resampled row indices."""
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
