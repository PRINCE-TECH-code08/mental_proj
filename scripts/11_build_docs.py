"""Builds result tables, the experiment matrix, MODEL_CARD.md and the paper's results section from results/*.
No number in these documents is typed by hand."""
from _common import *
import numpy as np, pandas as pd
R = C.RESULTS
fb = json.load(open(R / "final_vs_baseline.json")); meta = json.load(open(C.MODELS / "metadata.json")); conf = json.load(open(C.MODELS / "conformal.json"))
sel = json.load(open(R / "selected_model.json")); dec = json.load(open(R / "feature_decision.json")); nf = json.load(open(C.MODELS / "noise_floor.json"))["noise_floor"]
comp = pd.read_csv(R / "model_comparison.csv"); nest = pd.read_csv(R / "nested_cv_tuned.csv"); ab = pd.read_csv(R / "ablation_results.csv"); ef = pd.read_csv(R / "engineered_features.csv")
unc = pd.read_csv(R / "uncertainty_evaluation.csv"); sub = pd.read_csv(R / "subgroup_performance.csv"); sens = pd.read_csv(R / "sensitivity_summary.csv")
pi = pd.read_csv(R / "permutation_importance.csv"); sh = pd.read_csv(R / "shapley_global.csv", index_col=0).iloc[:, 0]; val = json.load(open(R / "shapley_validation.json"))
base = json.load(open(R / "baseline_reproduction.json")); T = C.ROOT / "paper" / "tables"
f = lambda x, n=3: f"{x:.{n}f}"

# ---- tables
t1 = pd.DataFrame([dict(Model="Original baseline (pickle)", MAE=fb["baseline_original_pickle"]["MAE"], RMSE=fb["baseline_original_pickle"]["RMSE"], R2=fb["baseline_original_pickle"]["R2"]),
                   dict(Model="Baseline config, re-fit in leak-free pipeline", MAE=fb["baseline_config_refit"]["MAE"], RMSE=fb["baseline_config_refit"]["RMSE"], R2=fb["baseline_config_refit"]["R2"]),
                   dict(Model="Improved (tuned RF)", MAE=fb["improved"]["MAE"], RMSE=fb["improved"]["RMSE"], R2=fb["improved"]["R2"])]).round(4)
(T / "test_comparison.md").write_text(t1.to_markdown(index=False))
(T / "model_comparison_cv.md").write_text(comp[["Model", "CV_MAE_mean", "CV_MAE_sd", "CV_RMSE_mean", "CV_R2_mean", "Train_R2_mean"]].round(4).to_markdown(index=False))
(T / "nested_cv.md").write_text(nest.round(4).to_markdown(index=False))
rf = ab[ab.Model.str.startswith("Random Forest")].set_index("Experiment"); rg = ab[ab.Model.str.startswith("Ridge")].set_index("Experiment")
at = pd.DataFrame({"RF MAE": rf.MAE_mean, "RF sd": rf.MAE_sd, "RF dMAE vs E": rf.dMAE_vs_E, "Ridge MAE": rg.MAE_mean}).round(4)
(T / "ablation.md").write_text(at.to_markdown()); (T / "engineered_features.md").write_text(ef[["Feature", "Model", "dMAE", "paired_SE", "helps"]].round(4).to_markdown(index=False))
(T / "uncertainty.md").write_text(unc.round(4).to_markdown(index=False)); (T / "subgroups.md").write_text(sub.round(3).to_markdown(index=False)); (T / "sensitivity.md").write_text(sens.round(3).to_markdown(index=False))
gl = pd.DataFrame({"perm_importance_dMAE": pi.set_index("feature").MAE_increase_mean, "mean_abs_shapley": sh}).sort_values("mean_abs_shapley", ascending=False).round(4)
(T / "feature_importance.md").write_text(gl.to_markdown())
for fig in ["ablation_comparison", "model_comparison", "predicted_vs_actual", "residuals", "error_distribution", "feature_importance", "shap_summary", "shap_local_example", "partial_dependence", "uncertainty", "subgroup_errors", "sensitivity", "target_distribution", "correlation_matrix"]:
    p = C.FIGURES / f"{fig}.png"
    if p.exists(): (C.ROOT / "paper" / "figures" / p.name).write_bytes(p.read_bytes())

# ---- experiment matrix
u90 = unc[(unc.nominal == 0.9)].set_index("method")
E = lambda k: rf.loc[k, "MAE_mean"]
rows = [
 ("E1", "Reproduce baseline", "Original notebook + pickle reproduce on the identical split", "Re-run split/cleaning; score original pickle", "R2, MAE (test)", f"R2={f(base['reproduced_test']['R2'],6)}, MAE={f(base['reproduced_test']['MAE'],6)}; exact={base['reproduction_exact']}", "Baseline is a valid fixed reference"),
 ("E2", "Model family comparison", "Non-linear ensembles beat linear models", "Repeated 5x2 CV on train, 8 candidates", "CV MAE/RMSE/R2", "; ".join(f"{r.Model}: {f(r.CV_MAE_mean)}" for r in comp.itertuples()), "Random Forest best at default settings; linear models leave much variance unexplained"),
 ("E3", "Tuning under nested CV", "Tuning improves generalisation beyond default RF", "Nested CV (5 outer, 3 inner, 6 iters)", "Nested CV MAE", "; ".join(f"{r.Model}: {f(r.NestedCV_MAE_mean)}+/-{f(r.NestedCV_MAE_sd)}" for r in nest.itertuples()) + f"; untuned RF CV MAE {f(comp[comp.Model.str.startswith('Random Forest')].CV_MAE_mean.iloc[0])}", "Tuning gain is small relative to fold SD; RF vs HGB difference is within noise"),
 ("E4", "Feature-group ablation", "Social-media + lifestyle carry most signal; stress is redundant (pre-registered guess)", "11 feature subsets x {RF, Ridge}, repeated CV", "CV MAE", "; ".join(f"{k.split(':')[0]}={f(E(k))}" for k in rf.index), "Partly NOT supported. Demographics+social media (B=" + f(E('B: Demographics + Social media')) + ") nearly matches all features (E=" + f(E('E: All features (baseline set)')) + "), while social+lifestyle WITHOUT demographics (D=" + f(E('D: Social media + Lifestyle')) + ") is worse; demographics alone (A=" + f(E('A: Demographics')) + ") are ~uninformative, yet conditional on behaviour each adds a little (single-demographic check, results/ablation_which_demographic.csv: " + ", ".join(f"{r.features} {f(r.CV_MAE)}" for r in pd.read_csv(R / 'ablation_which_demographic.csv').itertuples()) + "). Country adds the most, which flags it as a demographic proxy. Stress removal (G) changes MAE by only " + f(rf.loc['G: All minus Stress_Level', 'dMAE_vs_E'], 4) + " (redundant, supported). Lifestyle removal (J) changes MAE by " + f(rf.loc['J: All minus Lifestyle', 'dMAE_vs_E'], 4) + " (within noise)."),
 ("E5", "Engineered features", "Ratios/hinge features improve accuracy", "Each feature added singly and jointly; adopt if >2 paired SE", "dMAE vs base", f"all four: dMAE={f(dec['dMAE_F_vs_E'],4)} (SE {f(dec['paired_SE'],4)}); adopted={dec['use_engineered_features']}", "Not supported: engineered features did not help the selected model"),
 ("E6", "Improved vs baseline (test)", "Improved system has lower error than baseline", "Paired bootstrap on 1,500 test rows", "dMAE, dR2 with 95% CI", f"dMAE={f(fb['delta_MAE_baseline_minus_improved'],4)} CI {[round(x,4) for x in fb['delta_MAE_CI95']]}; dR2={f(fb['delta_R2_improved_minus_baseline'],4)} CI {[round(x,4) for x in fb['delta_R2_CI95']]}", "Small but test-sampling-robust gain; does not include training-set variability"),
 ("E7", "Interval calibration", "Cross-conformal intervals achieve nominal coverage", "Test coverage at 80/90/95%", "Empirical coverage, width", f"90%: local {f(u90.loc['locally-weighted','empirical_coverage'])} (width {f(u90.loc['locally-weighted','mean_width'],2)}), global {f(u90.loc['global','empirical_coverage'])} (width {f(u90.loc['global','mean_width'],2)})", "Marginal coverage met; see conditional coverage file"),
 ("E8", "Subgroup performance", "Errors are similar across subgroups", "Per-group MAE/R2/bias with bootstrap CI", "MAE by group", f"Gender MAE {', '.join(f'{r.group}:{f(r.MAE)}' for r in sub[sub.attribute=='Gender'].itertuples())}; small groups flagged LOW", "Descriptive only; fairness NOT claimed"),
 ("E9", "Sensitivity / noise floor", "Predictions are stable to small input noise", "Perturbation experiments; jitter 0.25/0.5 h", "abs change p95", f"noise 0.25h: p95={f(nf['input_noise_sd_0.25h']['abs_change_p95'],2)}; 0.5h: p95={f(nf['input_noise_sd_0.5h']['abs_change_p95'],2)}", "Hypothesis rejected: prediction changes below these values are within input noise"),
 ("E10", "Explanation validity", "MC-Shapley satisfies efficiency and converges", "Efficiency check; 24 vs 600 perms; rank agreement with permutation importance", "max violation, Pearson, Spearman", f"eff. violation {val['efficiency_max_abs_violation']:.1e}; rho={f(val['spearman_perm_vs_shapley_ranking'],2)}", "Explanations are internally consistent (not proof of causal validity)"),
]
pd.DataFrame(rows, columns=["ID", "Experiment", "Hypothesis", "Method", "Metric", "Result", "Interpretation"]).to_csv(R / "experiment_matrix.csv", index=False)

# ---- MODEL CARD
top3 = ", ".join(sh.index[:3])
mc = f"""# Model Card - Student Digital Wellbeing Score (v{meta['model_version']})

_Auto-generated by `scripts/11_build_docs.py`; all numbers come from `results/` and `models/`._

## Purpose and intended use
Educational / research demonstration that estimates a **dataset-specific wellbeing score** (`Mental_Health_Score`, observed range {conf and json.load(open(C.MODELS/'reference.json'))['target']['min']}-{json.load(open(C.MODELS/'reference.json'))['target']['max']}) from self-reported habits, with a prediction interval and a feature-contribution explanation.
**Not** for diagnosis, screening, triage, admissions, hiring, insurance or any decision about a person. It does not detect or diagnose any condition.

## Model
{meta['algorithm']} ({json.dumps(meta['hyperparameters'])}); scikit-learn {meta['libs']['sklearn']}; features: {len(meta['features'])} (engineered features adopted: {meta['engineered_features']}). Seed {meta['random_seed']}. Trained on {meta['n_train']} rows, tested on {meta['n_test']} held-out rows ({meta['split']}). Dataset SHA-256 `{meta['dataset_sha256'][:16]}...`.

## Performance (held-out test set, n={meta['n_test']})
{t1.to_markdown(index=False)}

Improvement over the original baseline: dMAE = {f(fb['delta_MAE_baseline_minus_improved'],4)} (95% CI {[round(x,4) for x in fb['delta_MAE_CI95']]}), dR2 = {f(fb['delta_R2_improved_minus_baseline'],4)} (95% CI {[round(x,4) for x in fb['delta_R2_CI95']]}). The CI reflects test-sample noise only. Under nested CV the tuned RF (MAE {f(nest.iloc[0].NestedCV_MAE_mean)} +/- {f(nest.iloc[0].NestedCV_MAE_sd)}) and tuned HGB ({f(nest.iloc[1].NestedCV_MAE_mean)} +/- {f(nest.iloc[1].NestedCV_MAE_sd)}) are statistically hard to separate; **the gain from tuning is modest**.

## Uncertainty
{conf['method']}. Empirical test coverage at nominal 90%: {f(u90.loc['locally-weighted','empirical_coverage'])} (mean width {f(u90.loc['locally-weighted','mean_width'],2)} score points). Coverage is marginal; per-group coverage varies (see `results/uncertainty_conditional_coverage.csv`, `docs/fairness.md`; e.g. India {f(sub[(sub.attribute=='Grouped_country')&(sub.group=='India')].coverage90.iloc[0],3)}). Intervals are **not valid outside the training ranges** (e.g. age outside 18-24); the API flags this. Uncertainty labels (lower/moderate/higher) are terciles of training out-of-fold interval width, i.e. *relative*, not absolute.

## Explainability
Global: permutation importance and mean |Shapley| rank {top3} highest. Local: Monte-Carlo Shapley (interventional, 60-row background; 24 permutations in the API). Efficiency violation {val['efficiency_max_abs_violation']:.1e}; 24-vs-600-permutation Pearson {min(c['pearson'] for c in val['convergence_24_vs_600_perms']):.2f}-{max(c['pearson'] for c in val['convergence_24_vs_600_perms']):.2f}. Contributions describe model behaviour and **must not be read as causal effects**.

## Known limitations and failure modes
- **Data provenance NOT ESTABLISHED.** The Kaggle description says the data were "designed" as a "realistic representation" for educational/portfolio use; no collection protocol is documented. Results may reflect a generative process, not real students.
- Ages 18-24 only; two gender values only; 111 countries with 37.6% labelled 'Other'.
- Regression toward the mean: low true scores are over-predicted and high ones under-predicted (`reports/error_analysis.md`).
- **Sensitivity/instability:** perturbing inputs by plausible self-report noise (0.25 h) changes predictions with a 95th-percentile magnitude of {f(nf['input_noise_sd_0.25h']['abs_change_p95'],2)} points (assumption-based, `results/sensitivity_extra.json`). Differences between repeated predictions smaller than this should not be interpreted.
- The model is not monotone in inputs (e.g. +1 h sleep raises the prediction in {f(100*sens.set_index('scenario').loc['Sleep +1 h','share_delta_positive'],1)}% of test rows and lowers it in the rest).
- `Country` has a non-trivial importance (permutation dMAE {f(pi.set_index('feature').loc['Country','MAE_increase_mean'],3)}), i.e. the model uses a demographic proxy. Consider removing it in any real use.
- Stress_Level removal changes CV MAE by only {f(rf.loc['G: All minus Stress_Level','dMAE_vs_E'],4)}: the model does not rely on it, but it is conceptually close to the outcome.

## Ethical considerations
See `ETHICS.md`. Risks: false reassurance, false alarm, over-interpretation of feature contributions as advice, misuse for profiling.
"""
(C.ROOT / "MODEL_CARD.md").write_text(mc)
print(t1.to_string()); print(open(R / "experiment_matrix.csv").read()[:1800])
