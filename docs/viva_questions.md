# Viva Preparation

Answers reference measured numbers where possible (`results/`, `MODEL_CARD.md`). Difficult/professor-level questions are marked **(hard)**.

## Machine Learning fundamentals
**Q: Why Random Forest over the tuned HistGradientBoosting, when their nested-CV MAEs (0.3487 vs 0.3573) are close?**
A: Random Forest had the lowest nested-CV MAE, and the difference is small relative to the fold standard deviations (0.019 vs 0.014), so I don't over-claim RF is definitively better -- I state the selection rule (lowest nested-CV MAE) explicitly and report both numbers rather than hiding the closeness.

**Q: What is the danger of the original notebook picking its final model by test-set score? (hard)**
A: It turns the test set into a second validation set; the reported test metric is then optimistic for whichever model/config was chosen, because you selected the one that happened to do best on that particular sample. My pipeline replaces this with nested cross-validation for selection and touches the test set exactly once, after the model is fixed.

**Q: Explain train_test_split leakage in this project's original country-grouping step.**
A: The original notebook computed the top-10 country list from the *full* cleaned dataframe, before splitting into train/test -- so information about the test set's country distribution technically informed the "Other" bucket used on training too. I verified the resulting top-10 set is identical whether computed on the full data or the training split alone here, so it didn't change results on this data, but I still moved the fit into the sklearn Pipeline so it is correct under resampling (e.g. cross-validation), where the leak would otherwise recur every fold.

**Q: Why nested CV specifically, not just CV then evaluate on test?**
A: Plain "CV to pick hyperparameters, then test once" is what I actually do for the *final* number. Nested CV is used additionally to get an *honest estimate of the tuning procedure's own generalisation*, without spending the real test set on it -- it simulates "what if we had many different train/test splits" using only the training data.

## Preprocessing / feature engineering
**Q: Why is Stress_Level's removal only costing 0.0025 MAE, when stress and wellbeing seem obviously related? (hard)**
A: Measured directly: `results/ablation_results.csv`, row G. Two explanations, not mutually exclusive: (1) the tree ensemble reconstructs similar information from the other 11 features (e.g. sleep, screen time), so Stress_Level is redundant rather than useless; (2) the association between self-reported stress and the target might be weaker in this specific dataset than intuition suggests. I don't claim which; I only report the measured redundancy.

**Q: Why didn't the engineered ratio features help?**
A: A tree-based model can already approximate ratios and thresholds via repeated splits on the raw features, so a hand-crafted ratio adds little beyond what's already learnable; the pre-registered rule (>2 paired SE improvement) formalizes "did it help enough to justify the complexity," and it didn't (`results/feature_decision.json`).

**Q: Why log1p on Study_Hours only?**
A: I kept exactly the original notebook's choice for comparability; a broader "which numeric features benefit from a log transform" study was not run and would need its own evidence before extending it.

## Explainability
**Q: Why Monte-Carlo Shapley instead of the SHAP package? (hard)**
A: The build sandbox had no network access, so `pip install shap` failed; I implemented the sampling-permutation method of Strumbelj & Kononenko (2014) instead, which is the same idea SHAP's `KernelExplainer`/`PermutationExplainer` approximate. I validated it two ways: efficiency (contributions sum exactly to prediction-minus-baseline, verified to floating-point precision) and convergence (24-permutation estimates correlate 0.98-0.99 with a 600-permutation reference on held-out instances).

**Q: Are Shapley values causal?**
A: No. They describe how the *fitted model's output* depends on each input, holding a background distribution fixed. They are not evidence that changing a feature would change a person's actual wellbeing.

## Uncertainty
**Q: What guarantee does your conformal interval actually have? (hard)**
A: Marginal coverage: averaged over the population the interval was calibrated and tested on, the true value falls inside the interval at roughly the nominal rate (measured: 92.27% at nominal 90% for the locally-weighted method). It does **not** guarantee coverage for every subgroup or every individual (Barber et al. 2021 prove this is impossible distribution-free); I checked conditional coverage empirically per subgroup and it does vary (`results/uncertainty_conditional_coverage.csv`), which I report rather than paper over.

**Q: Why cross-conformal / OOF residuals rather than a simple train-then-calibrate-on-test split?**
A: Calibrating on the test set would contaminate the one number (test performance) I want to remain honest; using out-of-fold residuals from the training data lets me calibrate without touching test data at all.

## Fairness
**Q: Is the model fair?**
A: I don't claim that. I report subgroup MAE/R2/bias with bootstrap CIs and flag small groups (n<30, n<100). Gender shows a small gap (0.323 vs 0.341 MAE); some country/platform groups show larger gaps but are underpowered. I explicitly avoid computing a single "fairness score" because the target is continuous, self-reported, and group sizes are too uneven for it to be meaningful.

**Q: What's missing to actually assess fairness properly? (hard)**
A: Measurement invariance evidence (does the score mean the same thing across groups), larger and more balanced samples per group, a stated fairness criterion agreed with stakeholders before evaluation (to avoid picking the criterion that looks best after the fact), and ideally a real user population rather than this dataset.

## Backend / API
**Q: How do you avoid the API crashing if the model file is missing or corrupted?**
A: Model loading happens in the FastAPI `lifespan` handler inside a try/except; on failure the app still starts, `app.state.service` is `None`, `/health` reports 503/degraded, and prediction endpoints raise a handled `RuntimeError` -> 503, instead of the process failing to boot.

**Q: Why keep the exact same `/predict` request/response shape as the original?**
A: Backward compatibility -- the improved API is a strict superset; anything that talked to the old API still works, and new fields (interval, explanation, warnings) are additive.

**Q: Why rate-limit and how does it work here?**
A: A simple in-memory sliding window per client IP in ASGI middleware; adequate for a small deployment but not distributed-safe (multiple server instances would each track their own window) -- documented as a known limitation, not sold as production-grade.

## Deployment
**Q: Why was Docker not actually built/tested?**
A: No Docker binary in the build sandbox. The Dockerfile follows standard practice (slim base, non-root user, `$PORT`/`0.0.0.0`) and is reviewed line-by-line; `.github/workflows/ci.yml` builds it on GitHub's runners, which is the actual verification step.

## Research methodology
**Q: What would you have done differently with more time/resources? (hard)**
A: Complete the systematic literature search before finalizing the research-gap claim; verify the Monte-Carlo Shapley values against the real `shap` package where network access allows; and most importantly, either establish this dataset's real provenance or replace it with a dataset that has a documented, consented collection process -- everything downstream inherits that uncertainty.

**Q: What's the single biggest limitation of this whole project?**
A: Dataset provenance. Every metric, interval, and subgroup finding is conditional on this specific 5,000-row file actually representing what it claims to; the source's own description ("designed... realistic representation... for educational use") does not establish that, so I treat every quantitative claim as scoped to this file rather than to real students.
