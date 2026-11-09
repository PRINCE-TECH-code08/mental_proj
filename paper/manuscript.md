# An Explainable, Uncertainty-Aware and Fairness-Evaluated System for Predicting a Student Digital-Wellbeing Score

## Abstract
We re-implement and extend a final-year student project that predicts a self-reported *Mental_Health_Score* (observed range 3.6-9.4) for 5,000 students from social-media, lifestyle and demographic features. Starting from a preserved, checksummed copy of the original notebook, pickle and web app, we build a leak-free scikit-learn pipeline, reproduce the original result exactly (test R2=0.878017, MAE=0.346502), and then evaluate model choice, feature groups and engineered features under repeated and nested cross-validation on the training split only. The selected model (tuned Random Forest) improves test-set MAE by 0.0138 points over the original model (95% bootstrap CI [0.0083, 0.019]), a small but test-sampling-robust gain. We add a cross-conformal uncertainty interval (locally-weighted, 92.3% empirical coverage at the 90% nominal level), a Monte-Carlo Shapley explanation validated for efficiency and convergence, subgroup error reporting with explicit sample-size caveats, and a noise-floor analysis showing that repeated predictions can change by 0.55 points from plausible self-report noise alone. We report an ablation finding that contradicts our own pre-registered hypothesis: `Stress_Level` is *not* the main driver of accuracy (removing it changes CV MAE by only 0.0025), while a demographic proxy (`Country`) contributes non-trivially. We do not claim the model is fair, causal, clinically valid, or that the dataset's provenance is established; all such claims are explicitly scoped as limitations.

**Keywords:** student wellbeing, digital wellbeing, explainable AI, conformal prediction, subgroup evaluation, reproducibility

## 1. Introduction
Concern about social media's effect on student mental health is long-standing and, in at least one quasi-experimental study, causally supported: Braghieri, Levy and Makarin (2022) use the staggered US-college rollout of Facebook and a difference-in-differences design to show a negative effect on student mental health, plausibly mediated by social comparison. Most classroom-scale ML projects, including the one this paper builds on, instead fit a supervised model to a single cross-sectional survey-style dataset and predict a wellbeing score from habits. Such models can be useful pedagogically and can surface associations, but the modelling choices that make them *trustworthy enough to report* -- leak-free preprocessing, model selection that does not peek at the test set, calibrated uncertainty, subgroup-aware error reporting, and an explicit accounting of what is not known about the data -- are frequently skipped. This paper documents doing that work end to end on one existing project, while being deliberately conservative about what the results do and do not show.

## 2. Related Work
Draženović et al. (2023) review 13 studies on social media and student/adolescent mental health during early COVID-19 and report a predominantly negative association, with two studies noting potential positive effects (support, connection); this motivates treating screen-time/wellbeing associations as directionally plausible but heterogeneous. Braghieri et al. (2022) provide causal evidence, in contrast to the purely observational, cross-sectional design used here and in the dataset's source project. For uncertainty quantification we use cross-conformal / split-conformal ideas from Lei, G'Sell, Rinaldo, Tibshirani and Wasserman (2018), who establish finite-sample marginal coverage for prediction bands built from an arbitrary regression estimator; related work by Barber, Candes, Ramdas and Tibshirani on the jackknife+ and on the limits of distribution-free *conditional* coverage explains why we report conditional coverage empirically rather than claim it is guaranteed. For local/global explanation we implement the sampling-based feature-contribution method of Strumbelj and Kononenko (2014), since the `shap` package could not be installed in our offline build environment; our implementation is validated against their method's defining property (efficiency: contributions sum exactly to the prediction gap) and against convergence with more permutations. A systematic literature search (PubMed/IEEE/ACM/Springer/arXiv) for closely related ML-on-this-dataset work was not completed within this session; `docs/literature_review.md` records what was verified and what remains open, and we make no claim of novelty.

## 3. Dataset
5,000 rows, 13 columns, no missing values (`reports/data_quality_report.md`). Cleaning removed 2 duplicate rows and clipped 10 negative `Physical_Activity_Hours` values to zero, leaving 4998 rows. Ages span 18-24 only; two gender categories only; 111 countries with 'Other' covering 970 rows. **Provenance is not established**: the dataset's own description states the records were "designed" as a "realistic representation" of student behaviour for educational/portfolio use, with no documented collection protocol (`DATA_CARD.md`). We treat all findings as findings about this file, not about students in general.

## 4. Methods
### 4.1 Pipeline and leakage control
All preprocessing (country top-10 grouping, scaling, encoding) is fit inside a scikit-learn `Pipeline`/`ColumnTransformer` on the training fold only (`src/mhp/features.py`). The original notebook instead computed the top-10 country list on the full cleaned frame before splitting; we verified the resulting top-10 country set is identical whether computed on the full frame or on the training split alone, so this particular leakage source does not change results on this dataset, but the leak-free version is used throughout so the pipeline is safe under resampling (e.g. cross-validation).

### 4.2 Model selection (train-only)
Eight candidates were compared with repeated 5x2 k-fold CV on the training split (`results/model_comparison.csv`): Random Forest (baseline config) (MAE 0.357), HistGradientBoosting (MAE 0.397), Decision Tree (MAE 0.441), Gradient Boosting (MAE 0.450), Linear Regression (MAE 0.524), Ridge (MAE 0.524), Lasso (MAE 0.537), Dummy (mean) (MAE 1.044). Random Forest and HistGradientBoosting were then tuned with `RandomizedSearchCV` inside a 5-fold nested CV to avoid optimistic bias (`results/nested_cv_tuned.csv`): Random Forest (tuned, nested CV): 0.349+/-0.019, HistGradientBoosting (tuned, nested CV): 0.357+/-0.014. Random Forest was selected (lowest nested-CV MAE); the gain from tuning over the untuned default (0.357) is small relative to fold-to-fold variability.

### 4.3 Feature-group ablation and engineered features
Eleven feature subsets were evaluated (`results/ablation_results.csv`, `figures/ablation_comparison.png`); paired differences use the same CV folds so a within-model paired standard error is reported. Four engineered ratio/hinge features (unlocks-per-screen-hour, sleep deficit, study/screen and activity/screen ratios) were tested singly and jointly against a pre-registered decision rule (adopt only if MAE improves by more than 2 paired SE); the rule was **not met** (0.0278 +/- 0.0011), so engineered features were **not** adopted in the final model (`results/feature_decision.json`).

### 4.4 Final evaluation against the original baseline
The final model (Random Forest, {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.33, "max_depth": null}) was fit once on the training split and evaluated once on the held-out test split, alongside the original pickled model scored on identical rows (Table 1). A paired bootstrap (2,000 resamples over the 1,500 test rows) gives the sampling-uncertainty of the improvement.

### 4.5 Uncertainty
Out-of-fold residuals from 5-fold CV on the training set calibrate a cross-conformal interval, in the spirit of Lei et al. (2018); a locally-weighted variant scales the interval by a secondary Random Forest's prediction of absolute residual (a simple heteroscedasticity model), floored at 0.05. Both variants are evaluated once on the test set for marginal and conditional coverage.

### 4.6 Explainability
Global: permutation importance (15 repeats) and mean |Shapley| over a 150-row test sample. Local: the same Monte-Carlo Shapley estimator (24 antithetic permutations at inference time; 200 for the illustrative example), using a 60-row training background. We check the efficiency property (contributions sum to prediction - baseline) and convergence of the 24-permutation estimate against a 600-permutation reference on 6 instances.

### 4.7 Subgroup and sensitivity analysis
Test-set MAE/RMSE/R2/bias/interval-coverage are reported per Gender, Academic_Level, Grouped_country, Most_Used_Platform and Stress_Level, with bootstrap CIs and explicit low-sample flags (n<30, n<100). We do not compute or claim any formal fairness metric (e.g. equalised odds) because the target is continuous and self-reported, and group sizes are uneven. Sensitivity experiments perturb one input at a time (e.g. sleep +/-1h) and are explicitly labelled as **model behaviour, not causal effects**; a separate noise-floor experiment jitters inputs by an assumed self-report error and reports how much predictions move as a result, to give a scale for interpreting repeated predictions in the frontend's trend view.

## 5. Results
### 5.1 Model comparison and selection
See `paper/tables/model_comparison_cv.md`, `paper/tables/nested_cv.md`, `paper/figures/model_comparison.png`.

### 5.2 Baseline vs improved (Table 1)
| Model                                         |    MAE |   RMSE |     R2 |
|:----------------------------------------------|-------:|-------:|-------:|
| Original baseline (pickle)                    | 0.3465 | 0.4629 | 0.878  |
| Baseline config, re-fit in leak-free pipeline | 0.3472 | 0.4637 | 0.8776 |
| Improved (tuned RF)                           | 0.3327 | 0.4421 | 0.8887 |

Improvement: dMAE = 0.0138 (95% CI [0.0083, 0.019]), dR2 = 0.0107 (95% CI [0.0067, 0.0152]). The CI excludes zero and reflects test-sample sampling noise only; it does not average over alternative training sets.

### 5.3 Ablation (Table 2, Figure 1)
| Experiment                     |   RF MAE |   RF sd |   RF dMAE vs E |   Ridge MAE |
|:-------------------------------|---------:|--------:|---------------:|------------:|
| A: Demographics                |   0.9803 |  0.0312 |         0.6334 |      1.0168 |
| B: Demographics + Social media |   0.3523 |  0.0116 |         0.0054 |      0.564  |
| C: Lifestyle                   |   0.5556 |  0.0182 |         0.2087 |      0.5735 |
| D: Social media + Lifestyle    |   0.3955 |  0.0149 |         0.0486 |      0.536  |
| E: All features (baseline set) |   0.3469 |  0.0108 |         0      |      0.5239 |
| F: All + engineered            |   0.3747 |  0.0116 |         0.0278 |      0.525  |
| G: All minus Stress_Level      |   0.3494 |  0.0121 |         0.0025 |      0.5258 |
| H: Stress_Level only           |   0.6223 |  0.019  |         0.2753 |      0.6212 |
| I: All minus Social media      |   0.419  |  0.0128 |         0.0721 |      0.5483 |
| J: All minus Lifestyle         |   0.3452 |  0.0123 |        -0.0017 |      0.5512 |
| K: All minus Demographics      |   0.3907 |  0.0143 |         0.0438 |      0.5345 |

Demographics+social-media alone (0.352) nearly matches the full feature set (0.347), while social-media+lifestyle *without* demographics is worse (0.396); a follow-up check (`results/ablation_which_demographic.csv`) shows `Country` contributes the most of the demographic features, flagging it as a proxy variable worth removing in any real deployment. Removing `Stress_Level` changes MAE by only 0.0025 -- **our pre-registered hypothesis that perceived stress dominates the prediction is not supported**.

### 5.4 Error analysis
See `reports/error_analysis.md`, `figures/predicted_vs_actual.png`, `figures/residuals.png`. The model exhibits regression-to-the-mean bias (over-predicts low true scores, under-predicts high ones).

### 5.5 Explainability (Figures 2-5)
Global importance ranks Avg_Daily_Usage_Hours, Sleep_Hours_Per_Night, Daily_Unlocks highest by both permutation importance and mean |Shapley| (Spearman rank correlation 0.88 between the two methods, `paper/tables/feature_importance.md`). The Monte-Carlo Shapley estimator's efficiency violation was 1.8e-15 (floating-point only) and its 24-permutation estimate correlated with a 600-permutation reference at Pearson 0.48-0.99 across 6 test instances.

### 5.6 Uncertainty calibration (Table 3, Figure 6)
| method           |   nominal |   empirical_coverage |   coverage_SE |   mean_width |   median_width |
|:-----------------|----------:|---------------------:|--------------:|-------------:|---------------:|
| global           |      0.8  |               0.816  |        0.01   |       1.103  |         1.103  |
| locally-weighted |      0.8  |               0.836  |        0.0096 |       1.1274 |         1.0982 |
| global           |      0.9  |               0.9193 |        0.007  |       1.546  |         1.546  |
| locally-weighted |      0.9  |               0.9227 |        0.0069 |       1.5186 |         1.4793 |
| global           |      0.95 |               0.958  |        0.0052 |       1.941  |         1.941  |
| locally-weighted |      0.95 |               0.9633 |        0.0049 |       1.8812 |         1.8326 |

The locally-weighted interval matches or exceeds nominal coverage at all three levels while producing narrower median widths than the global interval; conditional coverage by subgroup and by uncertainty label is in `results/uncertainty_conditional_coverage.csv`.

### 5.7 Subgroup performance (Table 4, Figure 7)
Gender MAE ranges from 0.323 (Female) to 0.341 (Male). Among country groups with n>=30, MAE ranges 0.239 (UK) to 0.376 (Canada); several country/platform groups have n<30 and are excluded from this comparison as underpowered (full table: `paper/tables/subgroups.md`). We explicitly do not conclude the model is fair.

### 5.8 Sensitivity and noise floor (Table 5, Figure 8)
| scenario                 |   mean_delta |   sd_delta |    p05 |   p95 |   share_delta_positive |   share_delta_negative |
|:-------------------------|-------------:|-----------:|-------:|------:|-----------------------:|-----------------------:|
| Sleep +1 h               |        0.175 |      0.289 | -0.197 | 0.738 |                  0.745 |                  0.255 |
| Sleep -1 h               |       -0.166 |      0.344 | -0.827 | 0.262 |                  0.385 |                  0.613 |
| Screen time +1 h         |       -0.215 |      0.305 | -0.742 | 0.217 |                  0.273 |                  0.725 |
| Screen time -1 h         |        0.233 |      0.297 | -0.168 | 0.763 |                  0.747 |                  0.25  |
| Physical activity +0.5 h |        0.011 |      0.086 | -0.12  | 0.153 |                  0.544 |                  0.449 |
| Study +1 h               |        0.016 |      0.135 | -0.2   | 0.234 |                  0.573 |                  0.426 |
| Unlocks +30              |       -0.074 |      0.234 | -0.472 | 0.314 |                  0.361 |                  0.639 |

Under an assumed self-report noise of 0.25h (resp. 0.5h) on the hour-valued inputs, predictions change with a 95th-percentile magnitude of 0.55 (resp. 0.76) score points (`results/sensitivity_extra.json`); the frontend's trend view uses this to warn when a session-to-session change is not distinguishable from noise. These are model-behaviour experiments and are not evidence that changing a habit would change a person's wellbeing.

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
