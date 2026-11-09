"""Phase 2-3: candidate comparison (repeated CV on TRAIN only), then nested-CV tuning of the serious candidates.
The test set is not touched here."""
import time
from _common import *
import numpy as np, pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate, RandomizedSearchCV
from mhp.features import make_pipeline

Xtr, Xte, ytr, yte = load_split()
SC = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=C.SEED)

cands = {
    "Dummy (mean)": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(random_state=C.SEED),
    "Random Forest (baseline config)": RandomForestRegressor(random_state=C.SEED),
    "Gradient Boosting": GradientBoostingRegressor(random_state=C.SEED),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=C.SEED),
}
rows = []
SKIP = (C.RESULTS / 'model_comparison.csv').exists()
for name, est in ([] if SKIP else cands.items()):
    t = time.time()
    r = cross_validate(make_pipeline(est), Xtr, ytr, cv=rkf, scoring=SC, return_train_score=True)
    row = {"Model": name, "fit_seconds": (time.time() - t) / rkf.get_n_splits()}
    for k in SC:
        v = -r[f"test_{k}"] if k != "R2" else r[f"test_{k}"]
        row[f"CV_{k}_mean"], row[f"CV_{k}_sd"] = v.mean(), v.std()
    row["Train_R2_mean"] = r["train_R2"].mean()
    row["Overfit_gap_R2"] = row["Train_R2_mean"] - row["CV_R2_mean"]
    rows.append(row); print(name, round(row["CV_MAE_mean"], 4), round(row["CV_R2_mean"], 4), flush=True)
comp = pd.read_csv(C.RESULTS / 'model_comparison.csv') if SKIP else pd.DataFrame(rows).sort_values('CV_MAE_mean')
comp.to_csv(C.RESULTS / 'model_comparison.csv', index=False)

# ---------- tuning -------------
spaces = {
    "Random Forest": (RandomForestRegressor(random_state=C.SEED),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 8, 12, 16, 24],
         "model__min_samples_leaf": [1, 2, 3, 5], "model__min_samples_split": [2, 5, 10],
         "model__max_features": [1.0, 0.7, 0.5, 0.33]}),
    "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=C.SEED),
        {"model__learning_rate": [.03, .05, .1, .2], "model__max_iter": [200, 400, 800],
         "model__max_leaf_nodes": [7, 15, 31], "model__min_samples_leaf": [10, 20, 40], "model__l2_regularization": [0, .1, 1]}),
}
NI = 6
outer = KFold(5, shuffle=True, random_state=C.SEED + 1)
nested, hp_rows, best = [], [], {}
import pickle
CACHE = C.RESULTS / '_tuning_cache'; CACHE.mkdir(exist_ok=True)
for name, (est, grid) in spaces.items():
    cf = CACHE / (name.replace(' ', '_') + '.pkl')
    if cf.exists():
        n_, h_, b_ = pickle.load(open(cf, 'rb')); nested.append(n_); hp_rows.append(h_); best[name] = b_; print('cached', name); continue
    t = time.time(); fold_mae, fold_r2 = [], []
    for k, (a, b) in enumerate(outer.split(Xtr)):
        s = RandomizedSearchCV(make_pipeline(est), grid, n_iter=NI, cv=3, scoring="neg_mean_absolute_error",
                               random_state=C.SEED, n_jobs=1)
        s.fit(Xtr.iloc[a], ytr.iloc[a])
        p = s.predict(Xtr.iloc[b]); e = ytr.iloc[b].values - p
        fold_mae.append(np.abs(e).mean()); fold_r2.append(1 - (e ** 2).sum() / ((ytr.iloc[b] - ytr.iloc[b].mean()) ** 2).sum())
    nested.append({"Model": name + " (tuned, nested CV)", "NestedCV_MAE_mean": np.mean(fold_mae), "NestedCV_MAE_sd": np.std(fold_mae),
                   "NestedCV_R2_mean": np.mean(fold_r2), "NestedCV_R2_sd": np.std(fold_r2), "seconds": time.time() - t})
    # final search on the full training set -> hyperparameters that will be used for the final model
    s = RandomizedSearchCV(make_pipeline(est), grid, n_iter=NI * 2, cv=5, scoring="neg_mean_absolute_error",
                           random_state=C.SEED, n_jobs=1, return_train_score=True).fit(Xtr, ytr)
    cr = pd.DataFrame(s.cv_results_)
    cr["Model"] = name
    hp_rows.append(cr[["Model", "params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]])
    best[name] = {k.replace("model__", ""): v for k, v in s.best_params_.items()}
    pickle.dump((nested[-1], hp_rows[-1], best[name]), open(cf, 'wb'))
    print(name, nested[-1], best[name], flush=True)
pd.DataFrame(nested).to_csv(C.RESULTS / "nested_cv_tuned.csv", index=False)
pd.concat(hp_rows).assign(cv_MAE=lambda d: -d.mean_test_score).to_csv(C.RESULTS / "hyperparameter_search.csv", index=False)

nd = pd.DataFrame(nested)
rf_base = comp[comp.Model == "Random Forest (baseline config)"].iloc[0]
w = nd.sort_values("NestedCV_MAE_mean").iloc[0]
sel = dict(selected=w.Model.replace(" (tuned, nested CV)", ""), params=best[w.Model.replace(" (tuned, nested CV)", "")],
           nested_cv_MAE=w.NestedCV_MAE_mean, nested_cv_MAE_sd=w.NestedCV_MAE_sd, all_best_params=best,
           rule="lowest nested-CV MAE on training data; test set untouched")
save_json(sel, C.RESULTS / "selected_model.json")

# figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3.8))
d = comp.iloc[::-1]
ax.barh(d.Model, d.CV_MAE_mean, xerr=d.CV_MAE_sd, color="#4C9A78"); ax.set_xlabel("Repeated 5x2 CV MAE (lower is better)")
ax.set_title("Model comparison (training data only)"); fig.savefig(C.FIGURES / "model_comparison.png"); plt.close(fig)
print(comp[["Model", "CV_MAE_mean", "CV_MAE_sd", "CV_R2_mean", "Train_R2_mean"]].round(4).to_string()); print(nd.round(4).to_string()); print(sel["selected"], sel["params"])
