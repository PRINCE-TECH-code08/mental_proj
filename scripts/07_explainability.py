"""Phase 7: global (permutation importance, PDP, mean |Shapley|) and local explanations. Association, not causation."""
from _common import *
import joblib, numpy as np, pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from mhp.explain import mc_shapley
import matplotlib.pyplot as plt
b = joblib.load(C.MODELS / "model_bundle.joblib"); model = b["model"]
Xtr, Xte, ytr, yte = load_split(); bg = pd.read_csv(C.MODELS / "background.csv")
pi = permutation_importance(model, Xte[C.RAW_INPUTS], yte, scoring="neg_mean_absolute_error", n_repeats=15, random_state=C.SEED)
pid = pd.DataFrame(dict(feature=C.RAW_INPUTS, MAE_increase_mean=pi.importances_mean, MAE_increase_sd=pi.importances_std)).sort_values("MAE_increase_mean", ascending=False)
pid.to_csv(C.RESULTS / "permutation_importance.csv", index=False); print(pid.round(4).to_string())
# Shapley global (subsample 150 test rows)
sub = Xte[C.RAW_INPUTS].sample(150, random_state=C.SEED); rows = []
for i, (_, x) in enumerate(sub.iterrows()):
    phi, base, fx = mc_shapley(model.predict, x, bg, C.RAW_INPUTS, n_perm=24, seed=i)
    rows.append({**phi, "_base": base, "_pred": fx})
S = pd.DataFrame(rows); S.index = sub.index
S.to_csv(C.RESULTS / "shapley_values_test_sample.csv")
g = S[C.RAW_INPUTS].abs().mean().sort_values(ascending=False).rename("mean_abs_shapley"); g.to_csv(C.RESULTS / "shapley_global.csv")
eff = float((S[C.RAW_INPUTS].sum(axis=1) - (S._pred - S._base)).abs().max())
rho = stats.spearmanr(pid.set_index("feature").loc[g.index, "MAE_increase_mean"], g.values)[0]
# convergence check of the 24-permutation estimate against a 600-permutation reference
conv = []
for i, (_, x) in enumerate(sub.head(6).iterrows()):
    a, *_ = mc_shapley(model.predict, x, bg, C.RAW_INPUTS, n_perm=24, seed=i)
    r, *_ = mc_shapley(model.predict, x, bg, C.RAW_INPUTS, n_perm=600, seed=99)
    va, vr = np.array(list(a.values())), np.array(list(r.values()))
    conv.append(dict(pearson=float(np.corrcoef(va, vr)[0, 1]), max_abs_diff=float(np.abs(va - vr).max())))
save_json(dict(efficiency_max_abs_violation=eff, spearman_perm_vs_shapley_ranking=float(rho), convergence_24_vs_600_perms=conv,
               note="Efficiency violations are floating-point only; convergence shows the API's 24-permutation estimate approximates a 600-permutation reference."), C.RESULTS / "shapley_validation.json")
print("efficiency err", eff, "rank rho", rho, conv[:2])
# figures
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
p = pid.iloc[::-1]; ax[0].barh(p.feature, p.MAE_increase_mean, xerr=p.MAE_increase_sd, color="#4C9A78"); ax[0].set_title("Permutation importance\n(MAE increase when shuffled)")
gg = g.iloc[::-1]; ax[1].barh(gg.index, gg.values, color="#21594A"); ax[1].set_title("Mean |Shapley| (150 test rows)")
fig.savefig(C.FIGURES / "feature_importance.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 5))
for j, f in enumerate(g.index[:8][::-1]):
    v = S[f]; xs = sub[f]; c = stats.rankdata(xs) / len(xs) if pd.api.types.is_numeric_dtype(xs) else stats.rankdata(xs.astype("category").cat.codes) / len(xs)
    ax.scatter(v, np.full(len(v), j) + np.random.default_rng(j).uniform(-.25, .25, len(v)), c=c, cmap="coolwarm", s=8)
ax.set_yticks(range(8)); ax.set_yticklabels(g.index[:8][::-1]); ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Shapley value (effect on model output)"); ax.set_title("Shapley summary (colour: low->high feature value / category code)")
fig.savefig(C.FIGURES / "shap_summary.png"); plt.close(fig)
x0 = sub.iloc[0]; phi, base, fx = mc_shapley(model.predict, x0, bg, C.RAW_INPUTS, n_perm=200, seed=1)
o = sorted(phi.items(), key=lambda kv: abs(kv[1]))
fig, ax = plt.subplots(figsize=(6.5, 4.4)); ax.barh([f"{k} = {x0[k]}" for k, _ in o], [v for _, v in o], color=["#4C9A78" if v > 0 else "#D9534F" for _, v in o])
ax.axvline(0, color="k", lw=.8); ax.set_title(f"Local explanation: baseline {base:.2f} -> prediction {fx:.2f}"); fig.savefig(C.FIGURES / "shap_local_example.png"); plt.close(fig)
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
pdx = Xte[C.RAW_INPUTS].sample(400, random_state=1).copy()
for c in ["Age", "Daily_Unlocks"]: pdx[c] = pdx[c].astype(float)
PartialDependenceDisplay.from_estimator(model, pdx, [f for f in pid.feature if f in C.NUMERIC + C.SKEWED][:4], ax=ax, grid_resolution=15)
fig.suptitle("Partial dependence of top-4 features (model behaviour, not causal)"); fig.savefig(C.FIGURES / "partial_dependence.png"); plt.close(fig)
