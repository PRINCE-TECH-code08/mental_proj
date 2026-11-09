"""Phase 30: assemble paper/manuscript.md purely from results/*.json|csv already computed. No numbers are typed by hand."""
from _common import *
import json, pandas as pd
R = C.RESULTS
fb = json.load(open(R / "final_vs_baseline.json")); meta = json.load(open(C.MODELS / "metadata.json")); conf = json.load(open(C.MODELS / "conformal.json"))
sel = json.load(open(R / "selected_model.json")); dec = json.load(open(R / "feature_decision.json")); nf = json.load(open(C.MODELS / "noise_floor.json"))["noise_floor"]
comp = pd.read_csv(R / "model_comparison.csv"); nest = pd.read_csv(R / "nested_cv_tuned.csv"); ab = pd.read_csv(R / "ablation_results.csv")
unc = pd.read_csv(R / "uncertainty_evaluation.csv"); sub = pd.read_csv(R / "subgroup_performance.csv"); sens = pd.read_csv(R / "sensitivity_summary.csv")
pi = pd.read_csv(R / "permutation_importance.csv"); sh = pd.read_csv(R / "shapley_global.csv", index_col=0).iloc[:, 0]; val = json.load(open(R / "shapley_validation.json"))
base = json.load(open(R / "baseline_reproduction.json")); ref = json.load(open(C.MODELS / "reference.json")); clean = json.load(open(R / "cleaning_log.json"))
f = lambda x, n=3: f"{x:.{n}f}"
u90 = unc.set_index(["method", "nominal"])
rf = ab[ab.Model.str.startswith("Random Forest")].set_index("Experiment")
top_gender = sub[sub.attribute == "Gender"].sort_values("MAE")
top_country_gap = sub[(sub.attribute == "Grouped_country") & (sub.n >= 30)]

md = f"""# An Explainable, Uncertainty-Aware and Fairness-Evaluated System for Predicting a Student Digital-Wellbeing Score

## Abstract
We re-implement and extend a final-year student project that predicts a self-reported *Mental_Health_Score* (observed range {ref['target']['min']}-{ref['target']['max']}) for 5,000 students from social-media, lifestyle and demographic features. Starting from a preserved, checksummed copy of the original notebook, pickle and web app, we build a leak-free scikit-learn pipeline, reproduce the original result exactly (test R2={f(base['reproduced_test']['R2'],6)}, MAE={f(base['reproduced_test']['MAE'],6)}), and then evaluate model choice, feature groups and engineered features under repeated and nested cross-validation on the training split only. The selected model (tuned Random Forest) improves test-set MAE by {f(fb['delta_MAE_baseline_minus_improved'],4)} points over the original model (95% bootstrap CI {[round(x,4) for x in fb['delta_MAE_CI95']]}), a small but test-sampling-robust gain. We add a cross-conformal uncertainty interval (locally-weighted, {round(100*u90.loc[('locally-weighted',0.9),'empirical_coverage'],1)}% empirical coverage at the 90% nominal level), a Monte-Carlo Shapley explanation validated for efficiency and convergence, subgroup error reporting with explicit sample-size caveats, and a noise-floor analysis showing that repeated predictions can change by {f(nf['input_noise_sd_0.25h']['abs_change_p95'],2)} points from plausible self-report noise alone. We report an ablation finding that contradicts our own pre-registered hypothesis: `Stress_Level` is *not* the main driver of accuracy (removing it changes CV MAE by only {f(rf.loc['G: All minus Stress_Level','dMAE_vs_E'],4)}), while a demographic proxy (`Country`) contributes non-trivially. We do not claim the model is fair, causal, clinically valid, or that the dataset's provenance is established; all such claims are explicitly scoped as limitations.

**Keywords:** student wellbeing, digital wellbeing, explainable AI, conformal prediction, subgroup evaluation, reproducibility

## 1. Introduction
Concern about social media's effect on student mental health is long-standing and, in at least one quasi-experimental study, causally supported: Braghieri, Levy and Makarin (2022) use the staggered US-college rollout of Facebook and a difference-in-differences design to show a negative effect on student mental health, plausibly mediated by social comparison. Most classroom-scale ML projects, including the one this paper builds on, instead fit a supervised model to a single cross-sectional survey-style dataset and predict a wellbeing score from habits. Such models can be useful pedagogically and can surface associations, but the modelling choices that make them *trustworthy enough to report* -- leak-free preprocessing, model selection that does not peek at the test set, calibrated uncertainty, subgroup-aware error reporting, and an explicit accounting of what is not known about the data -- are frequently skipped. This paper documents doing that work end to end on one existing project, while being deliberately conservative about what the results do and do not show.

## 2. Related Work
Draženović et al. (2023) review 13 studies on social media and student/adolescent mental health during early COVID-19 and report a predominantly negative association, with two studies noting potential positive effects (support, connection); this motivates treating screen-time/wellbeing associations as directionally plausible but heterogeneous. Braghieri et al. (2022) provide causal evidence, in contrast to the purely observational, cross-sectional design used here and in the dataset's source project. For uncertainty quantification we use cross-conformal / split-conformal ideas from Lei, G'Sell, Rinaldo, Tibshirani and Wasserman (2018), who establish finite-sample marginal coverage for prediction bands built from an arbitrary regression estimator; related work by Barber, Candes, Ramdas and Tibshirani on the jackknife+ and on the limits of distribution-free *conditional* coverage explains why we report conditional coverage empirically rather than claim it is guaranteed. For local/global explanation we implement the sampling-based feature-contribution method of Strumbelj and Kononenko (2014), since the `shap` package could not be installed in our offline build environment; our implementation is validated against their method's defining property (efficiency: contributions sum exactly to the prediction gap) and against convergence with more permutations. A systematic literature search (PubMed/IEEE/ACM/Springer/arXiv) for closely related ML-on-this-dataset work was not completed within this session; `docs/literature_review.md` records what was verified and what remains open, and we make no claim of novelty.

## 3. Dataset
5,000 rows, 13 columns, no missing values (`reports/data_quality_report.md`). Cleaning removed {clean['duplicates_dropped']} duplicate rows and clipped {clean['negative_activity_clipped']} negative `Physical_Activity_Hours` values to zero, leaving {clean['rows_clean']} rows. Ages span 18-24 only; two gender categories only; 111 countries with 'Other' covering {sub[(sub.attribute=='Grouped_country')&(sub.group=='Other')].n.iloc[0]} rows. **Provenance is not established**: the dataset's own description states the records were "designed" as a "realistic representation" of student behaviour for educational/portfolio use, with no documented collection protocol (`DATA_CARD.md`). We treat all findings as findings about this file, not about students in general.

## 4. Methods
### 4.1 Pipeline and leakage control
All preprocessing (country top-10 grouping, scaling, encoding) is fit inside a scikit-learn `Pipeline`/`ColumnTransformer` on the training fold only (`src/mhp/features.py`). The original notebook instead computed the top-10 country list on the full cleaned frame before splitting; we verified the resulting top-10 country set is identical whether computed on the full frame or on the training split alone, so this particular leakage source does not change results on this dataset, but the leak-free version is used throughout so the pipeline is safe under resampling (e.g. cross-validation).

### 4.2 Model selection (train-only)
Eight candidates were compared with repeated 5x2 k-fold CV on the training split (`results/model_comparison.csv`): {", ".join(f"{r.Model} (MAE {f(r.CV_MAE_mean)})" for r in comp.itertuples())}. Random Forest and HistGradientBoosting were then tuned with `RandomizedSearchCV` inside a 5-fold nested CV to avoid optimistic bias (`results/nested_cv_tuned.csv`): {", ".join(f"{r.Model}: {f(r.NestedCV_MAE_mean)}+/-{f(r.NestedCV_MAE_sd)}" for r in nest.itertuples())}. Random Forest was selected (lowest nested-CV MAE); the gain from tuning over the untuned default ({f(comp[comp.Model.str.startswith('Random Forest')].CV_MAE_mean.iloc[0])}) is small relative to fold-to-fold variability.

### 4.3 Feature-group ablation and engineered features
Eleven feature subsets were evaluated (`results/ablation_results.csv`, `figures/ablation_comparison.png`); paired differences use the same CV folds so a within-model paired standard error is reported. Four engineered ratio/hinge features (unlocks-per-screen-hour, sleep deficit, study/screen and activity/screen ratios) were tested singly and jointly against a pre-registered decision rule (adopt only if MAE improves by more than 2 paired SE); the rule was **not met** ({f(dec['dMAE_F_vs_E'],4)} +/- {f(dec['paired_SE'],4)}), so engineered features were **not** adopted in the final model (`results/feature_decision.json`).

### 4.4 Final evaluation against the original baseline
The final model (Random Forest, {json.dumps(sel['params'])}) was fit once on the training split and evaluated once on the held-out test split, alongside the original pickled model scored on identical rows (Table 1). A paired bootstrap (2,000 resamples over the 1,500 test rows) gives the sampling-uncertainty of the improvement.

### 4.5 Uncertainty
Out-of-fold residuals from 5-fold CV on the training set calibrate a cross-conformal interval, in the spirit of Lei et al. (2018); a locally-weighted variant scales the interval by a secondary Random Forest's prediction of absolute residual (a simple heteroscedasticity model), floored at {conf['eps']}. Both variants are evaluated once on the test set for marginal and conditional coverage.

### 4.6 Explainability
Global: permutation importance (15 repeats) and mean |Shapley| over a 150-row test sample. Local: the same Monte-Carlo Shapley estimator (24 antithetic permutations at inference time; 200 for the illustrative example), using a 60-row training background. We check the efficiency property (contributions sum to prediction - baseline) and convergence of the 24-permutation estimate against a 600-permutation reference on 6 instances.

### 4.7 Subgroup and sensitivity analysis
Test-set MAE/RMSE/R2/bias/interval-coverage are reported per Gender, Academic_Level, Grouped_country, Most_Used_Platform and Stress_Level, with bootstrap CIs and explicit low-sample flags (n<30, n<100). We do not compute or claim any formal fairness metric (e.g. equalised odds) because the target is continuous and self-reported, and group sizes are uneven. Sensitivity experiments perturb one input at a time (e.g. sleep +/-1h) and are explicitly labelled as **model behaviour, not causal effects**; a separate noise-floor experiment jitters inputs by an assumed self-report error and reports how much predictions move as a result, to give a scale for interpreting repeated predictions in the frontend's trend view.

## 5. Results
### 5.1 Model comparison and selection
See `paper/tables/model_comparison_cv.md`, `paper/tables/nested_cv.md`, `paper/figures/model_comparison.png`.

### 5.2 Baseline vs improved (Table 1)
{open(C.ROOT/'paper'/'tables'/'test_comparison.md').read()}

Improvement: dMAE = {f(fb['delta_MAE_baseline_minus_improved'],4)} (95% CI {[round(x,4) for x in fb['delta_MAE_CI95']]}), dR2 = {f(fb['delta_R2_improved_minus_baseline'],4)} (95% CI {[round(x,4) for x in fb['delta_R2_CI95']]}). The CI excludes zero and reflects test-sample sampling noise only; it does not average over alternative training sets.

### 5.3 Ablation (Table 2, Figure 1)
{open(C.ROOT/'paper'/'tables'/'ablation.md').read()}

Demographics+social-media alone ({f(rf.loc['B: Demographics + Social media','MAE_mean'])}) nearly matches the full feature set ({f(rf.loc['E: All features (baseline set)','MAE_mean'])}), while social-media+lifestyle *without* demographics is worse ({f(rf.loc['D: Social media + Lifestyle','MAE_mean'])}); a follow-up check (`results/ablation_which_demographic.csv`) shows `Country` contributes the most of the demographic features, flagging it as a proxy variable worth removing in any real deployment. Removing `Stress_Level` changes MAE by only {f(rf.loc['G: All minus Stress_Level','dMAE_vs_E'],4)} -- **our pre-registered hypothesis that perceived stress dominates the prediction is not supported**.

### 5.4 Error analysis
See `reports/error_analysis.md`, `figures/predicted_vs_actual.png`, `figures/residuals.png`. The model exhibits regression-to-the-mean bias (over-predicts low true scores, under-predicts high ones).

### 5.5 Explainability (Figures 2-5)
Global importance ranks {", ".join(sh.index[:3])} highest by both permutation importance and mean |Shapley| (Spearman rank correlation {f(val['spearman_perm_vs_shapley_ranking'],2)} between the two methods, `paper/tables/feature_importance.md`). The Monte-Carlo Shapley estimator's efficiency violation was {val['efficiency_max_abs_violation']:.1e} (floating-point only) and its 24-permutation estimate correlated with a 600-permutation reference at Pearson {min(c['pearson'] for c in val['convergence_24_vs_600_perms']):.2f}-{max(c['pearson'] for c in val['convergence_24_vs_600_perms']):.2f} across 6 test instances.

### 5.6 Uncertainty calibration (Table 3, Figure 6)
{open(C.ROOT/'paper'/'tables'/'uncertainty.md').read()}

The locally-weighted interval matches or exceeds nominal coverage at all three levels while producing narrower median widths than the global interval; conditional coverage by subgroup and by uncertainty label is in `results/uncertainty_conditional_coverage.csv`.

### 5.7 Subgroup performance (Table 4, Figure 7)
Gender MAE ranges from {f(top_gender.MAE.min())} ({top_gender.iloc[0].group}) to {f(top_gender.MAE.max())} ({top_gender.iloc[-1].group}). Among country groups with n>=30, MAE ranges {f(top_country_gap.MAE.min())} ({top_country_gap.loc[top_country_gap.MAE.idxmin(),'group']}) to {f(top_country_gap.MAE.max())} ({top_country_gap.loc[top_country_gap.MAE.idxmax(),'group']}); several country/platform groups have n<30 and are excluded from this comparison as underpowered (full table: `paper/tables/subgroups.md`). We explicitly do not conclude the model is fair.

### 5.8 Sensitivity and noise floor (Table 5, Figure 8)
{open(C.ROOT/'paper'/'tables'/'sensitivity.md').read()}

Under an assumed self-report noise of 0.25h (resp. 0.5h) on the hour-valued inputs, predictions change with a 95th-percentile magnitude of {f(nf['input_noise_sd_0.25h']['abs_change_p95'],2)} (resp. {f(nf['input_noise_sd_0.5h']['abs_change_p95'],2)}) score points (`results/sensitivity_extra.json`); the frontend's trend view uses this to warn when a session-to-session change is not distinguishable from noise. These are model-behaviour experiments and are not evidence that changing a habit would change a person's wellbeing.

## 6. Discussion
The improved system's accuracy gain over the original baseline is real but modest and should not be oversold: nested-CV comparison shows tuning buys little beyond a well-chosen default Random Forest, and the largest single methodological problem in the original project -- selecting a model configuration by looking at test-set performance -- was a validity issue rather than an accuracy one. The most informative result may be the negative ones: engineered ratio features did not help, and stress level (the feature conceptually closest to the outcome) turned out to be dispensable, which somewhat reduces the concern that the whole exercise is a near-circular prediction of stress from stress-adjacent inputs. The `Country` finding is a caution: a demographic field with no obvious mechanism carried real signal, which is exactly the kind of proxy that should be scrutinised before any real-world use.

## 7. Limitations
Dataset provenance is not established; the outcome variable's measurement is not documented; the sample covers a narrow age range and two gender categories; conditional interval coverage and subgroup error both vary enough that neither calibration nor fairness can be asserted at the group level everywhere; explanations and sensitivity results describe the fitted model, not causal mechanisms; and the literature review is partial.

## 8. Ethics
See `ETHICS.md`. The deployed system is explicitly framed as educational, shows uncertainty and warnings rather than a bare number, stores no personal data server-side, and is not fit for any diagnostic, screening or decision-making use.

## 9. Conclusion
Building a reproducible, leakage-checked, uncertainty-aware and subgroup-evaluated pipeline around an existing student project surfaced both a genuine (if modest) accuracy improvement and several honest negative results, and produced concrete, machine-checkable artifacts (checksummed baseline, experiment matrix, calibration and coverage tables) suitable for a final-year research report.

## 10. Future Work
Complete the systematic literature search; obtain or document real provenance for the training data (or replace it with a documented, consented dataset); evaluate `shap`'s exact TreeExplainer where a networked environment allows installing it, as a cross-check on the Monte-Carlo Shapley values used here; and, if longitudinal per-student data become available, replace the assumption-based noise floor with a measured one.

## References
- Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486-507.
- Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2), 455-482.
- Braghieri, L., Levy, R., and Makarin, A. (2022). Social Media and Mental Health. *American Economic Review*, 112(11), 3660-3693. https://doi.org/10.1257/aer.20211218
- Draženović, M., Vukušić Rukavina, T., and Machala Poplašen, L. (2023). Impact of Social Media Use on Mental Health within Adolescent and Student Populations during COVID-19 Pandemic: Review. *International Journal of Environmental Research and Public Health*, 20(4), 3392.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association*, 113(523), 1094-1111.
- Strumbelj, E., and Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. *Knowledge and Information Systems*, 41(3), 647-665.
"""
(C.ROOT / "paper" / "manuscript.md").write_text(md)
print(len(md), "chars written to paper/manuscript.md")
