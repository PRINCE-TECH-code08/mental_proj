"""Phase 4-5: feature-group ablation + evidence for each engineered feature. Repeated 5x2 CV on TRAIN only.
Uses the model selected in step 03 (and Ridge as an interpretable reference)."""
import json
from _common import *
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from mhp.features import make_pipeline
from mhp.models import make_estimator

Xtr, Xte, ytr, yte = load_split()
sel = json.load(open(C.RESULTS / "selected_model.json"))
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=C.SEED)
G = C.GROUPS
EXPS = {
    "A: Demographics": G["demographics"],
    "B: Demographics + Social media": G["demographics"] + G["social_media"],
    "C: Lifestyle": G["lifestyle"],
    "D: Social media + Lifestyle": G["social_media"] + G["lifestyle"],
    "E: All features (baseline set)": C.BASE_FEATURES,
    "F: All + engineered": C.BASE_FEATURES + C.ENGINEERED,
    "G: All minus Stress_Level": [c for c in C.BASE_FEATURES if c not in G["stress"]],
    "H: Stress_Level only": G["stress"],
    "I: All minus Social media": [c for c in C.BASE_FEATURES if c not in G["social_media"]],
    "J: All minus Lifestyle": [c for c in C.BASE_FEATURES if c not in G["lifestyle"]],
    "K: All minus Demographics": [c for c in C.BASE_FEATURES if c not in G["demographics"]],
}
MODELS = {sel["selected"] + " (selected)": lambda: make_estimator(sel["selected"], sel["params"]),
          "Ridge (reference)": lambda: Ridge(alpha=1.0)}


def cv(est, feats):
    eng = any(f in C.ENGINEERED for f in feats)
    r = cross_validate(make_pipeline(est, feats, engineered=eng), Xtr, ytr, cv=rkf,
                       scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"})
    return -r["test_MAE"], -r["test_RMSE"], r["test_R2"]


rows, fold_mae = [], {}
for mname, mk in MODELS.items():
    for ename, feats in EXPS.items():
        mae, rmse, r2 = cv(mk(), feats); fold_mae[(mname, ename)] = mae
        rows.append(dict(Model=mname, Experiment=ename, n_features=len(feats), MAE_mean=mae.mean(), MAE_sd=mae.std(),
                         RMSE_mean=rmse.mean(), R2_mean=r2.mean(), R2_sd=r2.std()))
        print(mname, ename, round(mae.mean(), 4), flush=True)
ab = pd.DataFrame(rows)
E_ = "E: All features (baseline set)"
ab["dMAE_vs_E"] = [r.MAE_mean - fold_mae[(r.Model, E_)].mean() for r in ab.itertuples()]
ab["dMAE_vs_E_paired_SE"] = [(fold_mae[(r.Model, r.Experiment)] - fold_mae[(r.Model, E_)]).std() / np.sqrt(len(fold_mae[(r.Model, E_)])) for r in ab.itertuples()]
ab.to_csv(C.RESULTS / "ablation_results.csv", index=False)

# --- engineered features one at a time
DEF = {
 "Unlocks_Per_Screen_Hour": ("Daily_Unlocks / (Avg_Daily_Usage_Hours + 0.1)", "Checking frequency independent of total screen time"),
 "Sleep_Deficit_Hours": ("max(0, 7 - Sleep_Hours_Per_Night)", "Shortfall below the lower bound of the commonly cited 7-9 h adult range (a hinge the trees/linear model could not otherwise express in one feature)"),
 "Study_Screen_Ratio": ("Study_Hours / (Avg_Daily_Usage_Hours + 0.1)", "Relative allocation of time to study vs screens"),
 "Activity_Screen_Ratio": ("Physical_Activity_Hours / (Avg_Daily_Usage_Hours + 0.1)", "Relative allocation of time to activity vs screens"),
}
ef = []
for mname, mk in MODELS.items():
    base = fold_mae[(mname, E_)]
    for f in C.ENGINEERED:
        mae, _, _ = cv(mk(), C.BASE_FEATURES + [f]); d = mae - base
        ef.append(dict(Feature=f, Definition=DEF[f][0], Reason=DEF[f][1], Model=mname, MAE_base=base.mean(), MAE_with=mae.mean(),
                       dMAE=d.mean(), paired_SE=d.std() / np.sqrt(len(d)), helps=bool(d.mean() < -2 * d.std() / np.sqrt(len(d)))))
        print(mname, f, round(d.mean(), 4), flush=True)
efd = pd.DataFrame(ef); efd.to_csv(C.RESULTS / "engineered_features.csv", index=False)

# --- decision rule (selected model only): adopt engineered set iff improvement > 2 paired SE
m0 = list(MODELS)[0]
d = fold_mae[(m0, "F: All + engineered")] - fold_mae[(m0, E_)]
use_eng = bool(d.mean() < -2 * d.std() / np.sqrt(len(d)))
dec = dict(model=m0, use_engineered_features=use_eng, dMAE_F_vs_E=d.mean(), paired_SE=d.std() / np.sqrt(len(d)),
           rule="adopt engineered features only if MAE improves by more than 2 paired standard errors")
save_json(dec, C.RESULTS / "feature_decision.json"); print(dec)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for a, (mname, g) in zip(ax, ab.groupby("Model", sort=False)):
    g = g.iloc[::-1]; cols = ["#C97C3B" if "G:" in e or "H:" in e else "#4C9A78" for e in g.Experiment]
    a.barh(g.Experiment, g.MAE_mean, xerr=g.MAE_sd, color=cols); a.set_title(mname, fontsize=10); a.set_xlabel("CV MAE (lower is better)")
fig.suptitle("Feature-group ablation (orange = stress-related contrasts)"); fig.savefig(C.FIGURES / "ablation_comparison.png"); plt.close(fig)
