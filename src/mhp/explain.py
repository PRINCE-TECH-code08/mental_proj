"""Model-agnostic local/global explanation.
`shap` could not be installed in the build environment, so we implement permutation-sampling Shapley values
(Strumbelj & Kononenko 2014) with an INTERVENTIONAL background: features are switched from a background row to the
instance one at a time along random orderings. Antithetic (reversed) orderings reduce variance. Each ordering
telescopes exactly, so   sum(contributions) == prediction - baseline   (efficiency) holds by construction.
Contributions describe how the MODEL's output depends on inputs; they are not causal effects."""
import numpy as np
import pandas as pd


def mc_shapley(predict, x: pd.Series, background: pd.DataFrame, features, n_perm=24, seed=0):
    rng = np.random.default_rng(seed)
    d = len(features)
    perms = np.array([rng.permutation(d) for _ in range(n_perm // 2)])
    perms = np.vstack([perms, perms[:, ::-1]])                      # antithetic pairs
    K = len(perms)
    bg = background.iloc[rng.integers(0, len(background), K)].reset_index(drop=True)
    pos = np.empty((K, d), int)
    for k in range(K):
        pos[k, perms[k]] = np.arange(d)
    cols = {}
    for j, f in enumerate(features):
        bvals = bg[f].to_numpy()
        steps = np.arange(d + 1)[None, :]                             # step t: features with pos < t come from x
        mask = pos[:, [j]] < steps                                    # (K, d+1)
        xv = np.full((K, d + 1), x[f], dtype=object)
        cols[f] = np.where(mask, xv, np.repeat(bvals[:, None], d + 1, 1)).reshape(-1)
    frame = pd.DataFrame(cols)
    for f in features:                                               # restore numeric dtypes
        if pd.api.types.is_numeric_dtype(background[f]):
            frame[f] = frame[f].astype(float)
    p = np.asarray(predict(frame)).reshape(K, d + 1)
    phi = np.zeros(d)
    for k in range(K):
        for t in range(d):
            phi[perms[k, t]] += p[k, t + 1] - p[k, t]
    phi /= K
    return dict(zip(features, phi)), float(p[:, 0].mean()), float(p[:, -1].mean())


def display_value(v):
    return round(float(v), 2) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
