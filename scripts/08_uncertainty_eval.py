"""Phase 8: evaluate interval calibration on the untouched test set (coverage, width, conditional coverage)."""
from _common import *
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
b = joblib.load(C.MODELS / "model_bundle.joblib"); conf = b["conformal"]
Xtr, Xte, ytr, yte = load_split()
pred = b["model"].predict(Xte); sig = np.maximum(b["sigma_model"].predict(Xte), conf["eps"]); y = yte.values; err = np.abs(y - pred)
rows = []
for a in conf["alphas"]:
    for kind, hw in (("global", np.full(len(y), conf["global_q"][str(a)])), ("locally-weighted", conf["normalized_q"][str(a)] * sig)):
        lo, hi = np.clip(pred - hw, 0, 10), np.clip(pred + hw, 0, 10); cov = (y >= lo) & (y <= hi)
        rows.append(dict(method=kind, nominal=1 - a, empirical_coverage=cov.mean(), coverage_SE=np.sqrt(cov.mean() * (1 - cov.mean()) / len(y)), mean_width=(hi - lo).mean(), median_width=np.median(hi - lo)))
U = pd.DataFrame(rows); U.to_csv(C.RESULTS / "uncertainty_evaluation.csv", index=False); print(U.round(4).to_string())
# conditional coverage at 90%
hw = conf["normalized_q"]["0.1"] * sig; cov = (np.abs(y - pred) <= hw); ghw = conf["global_q"]["0.1"]; gcov = np.abs(y - pred) <= ghw
cc = []
for col in ["Stress_Level", "Gender", "Academic_Level"]:
    for k, ix in Xte.reset_index(drop=True).groupby(col).groups.items():
        cc.append(dict(group=f"{col}={k}", n=len(ix), coverage_local=cov[ix].mean(), coverage_global=gcov[ix].mean(), width_local=2 * hw[ix].mean(), width_global=2 * ghw))
t1, t2 = conf["uncertainty_thresholds_halfwidth90"]; lab = np.where(hw <= t1, "lower", np.where(hw <= t2, "moderate", "higher"))
for k in ("lower", "moderate", "higher"):
    m = lab == k; cc.append(dict(group=f"uncertainty label={k}", n=int(m.sum()), coverage_local=cov[m].mean(), coverage_global=gcov[m].mean(), width_local=2 * hw[m].mean(), width_global=2 * ghw, MAE=err[m].mean()))
pd.DataFrame(cc).to_csv(C.RESULTS / "uncertainty_conditional_coverage.csv", index=False); print(pd.DataFrame(cc).round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for kind, c in (("global", "#C97C3B"), ("locally-weighted", "#21594A")):
    d = U[U.method == kind]; ax[0].plot(d.nominal, d.empirical_coverage, "o-", label=kind, color=c)
ax[0].plot([.75, 1], [.75, 1], "k--", lw=1); ax[0].set_xlabel("nominal coverage"); ax[0].set_ylabel("empirical (test)"); ax[0].legend(); ax[0].set_title("Calibration")
sel = pd.DataFrame(cc); sl = sel[sel.group.str.startswith("uncertainty")]; ax[1].bar(sl.group.str.split("=").str[1], sl.MAE, color="#4C9A78"); ax[1].set_title("Actual MAE by uncertainty label"); ax[1].set_ylabel("MAE")
fig.savefig(C.FIGURES / "uncertainty.png"); plt.close(fig)
