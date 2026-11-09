"""Phase 10-11: MODEL SENSITIVITY experiments (not causal) + noise floor for interpreting trends in repeated predictions."""
from _common import *
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
b = joblib.load(C.MODELS / "model_bundle.joblib"); model = b["model"]; ref = b["reference"]["numeric"]
Xtr, Xte, ytr, yte = load_split(); X = Xte[C.RAW_INPUTS].reset_index(drop=True); base = model.predict(X)
def shift(col, d):
    Z = X.copy(); Z[col] = np.clip(Z[col] + d, ref[col]["min"], ref[col]["max"]); return model.predict(Z) - base
scen = [("Sleep +1 h", "Sleep_Hours_Per_Night", 1), ("Sleep -1 h", "Sleep_Hours_Per_Night", -1), ("Screen time +1 h", "Avg_Daily_Usage_Hours", 1),
        ("Screen time -1 h", "Avg_Daily_Usage_Hours", -1), ("Physical activity +0.5 h", "Physical_Activity_Hours", .5), ("Study +1 h", "Study_Hours", 1), ("Unlocks +30", "Daily_Unlocks", 30)]
rows = []; D_ = {}
for n, c, d in scen:
    v = shift(c, d); D_[n] = v
    rows.append(dict(scenario=n, mean_delta=v.mean(), sd_delta=v.std(), p05=np.percentile(v, 5), p95=np.percentile(v, 95), share_delta_positive=(v > 1e-9).mean(), share_delta_negative=(v < -1e-9).mean()))
S = pd.DataFrame(rows); S.to_csv(C.RESULTS / "sensitivity_summary.csv", index=False); print(S.round(3).to_string())
# stress-level counterfactual sweep (model behaviour only)
sw = {}
for lv in C.STRESS_ORDER:
    Z = X.copy(); Z["Stress_Level"] = lv; sw[lv] = float(model.predict(Z).mean())
# noise floor: how much does the prediction move when inputs are jittered by plausible self-report error?
rng = np.random.default_rng(C.SEED); nf = {}
for sd_h in (0.25, 0.5):
    d = []
    for _ in range(10):
        Z = X.copy()
        for c in ["Avg_Daily_Usage_Hours", "Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]:
            Z[c] = np.clip(Z[c] + rng.normal(0, sd_h, len(Z)), ref[c]["min"], ref[c]["max"])
        Z["Daily_Unlocks"] = np.clip(Z["Daily_Unlocks"] * rng.normal(1, .1, len(Z)), ref["Daily_Unlocks"]["min"], ref["Daily_Unlocks"]["max"])
        d.append(model.predict(Z) - base)
    d = np.concatenate(d); nf[f"input_noise_sd_{sd_h}h"] = dict(prediction_change_sd=float(d.std()), abs_change_p95=float(np.percentile(np.abs(d), 95)))
save_json(dict(mean_prediction_by_forced_stress_level=sw, noise_floor=nf,
               assumption="Self-report error SD of 0.25/0.5 h (hours variables) and 10% (unlocks) is an ASSUMPTION for illustration; real self-report error was NOT MEASURED.",
               interpretation="Changes in the predicted score between two sessions that are smaller than abs_change_p95 are within what input noise alone can produce under these assumptions. A trend in model predictions does not establish a clinical change."), C.RESULTS / "sensitivity_extra.json")
save_json(dict(noise_floor=nf, assumption="Assumed self-report noise (0.25/0.5 h; 10% unlocks). NOT measured."), C.MODELS / "noise_floor.json")
print(sw, nf)
fig, ax = plt.subplots(figsize=(7, 3.8)); ax.boxplot([D_[n] for n, *_ in scen], vert=False, labels=[n for n, *_ in scen], showfliers=False)
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("change in predicted score"); ax.set_title("Model sensitivity experiments (NOT causal effects)"); fig.savefig(C.FIGURES / "sensitivity.png"); plt.close(fig)
