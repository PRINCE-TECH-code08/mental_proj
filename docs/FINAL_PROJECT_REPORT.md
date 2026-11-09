# Final-Year Project Report

**Title:** An Explainable, Uncertainty-Aware and Fairness-Evaluated System for Predicting a Student Digital-Wellbeing Score
**Certificate / Declaration / Acknowledgement:** [placeholder -- insert your institution's required text and signatures]
**Student(s):** [add] · **Supervisor:** [add] · **Institution:** [add] · **Department:** [add] · **Academic year:** [add]

---

## Abstract
See `paper/manuscript.md` (Abstract section) -- reused verbatim here to keep the two documents consistent.

## 1. Introduction
This project starts from an existing final-year-style application that predicts a student wellbeing score from social-media and lifestyle habits (preserved, unmodified, in `baseline/`), audits it, and rebuilds a research-grade pipeline and full-stack system around it without discarding the original work. Motivation and related work are given in `paper/manuscript.md` Sections 1-2.

## 2. Literature Review
`docs/literature_review.md` (partial; entries are verified, not fabricated; no novelty claim is made pending a full systematic search).

## 3. Methodology
`paper/manuscript.md` Section 4, in full detail: leak-free pipeline (`src/mhp/features.py`), model selection via repeated and nested cross-validation (`scripts/03_model_selection.py`), feature-group ablation (`scripts/04_ablation_and_features.py`), final training and calibration (`scripts/05_train_final.py`), error analysis, explainability, uncertainty evaluation, subgroup analysis and sensitivity analysis (`scripts/06`-`10`).

## 4. Implementation
- **Data & ML core:** `src/mhp/` (config, data, features, models, metrics, explain, service) -- framework-free, unit-tested (`tests/test_pipeline_and_service.py`, 14 tests, all passing).
- **Backend:** `backend/app.py`, FastAPI, backward-compatible with the original `/predict` contract, adds `/health`, `/model-info`, `/predict-with-uncertainty`, `/explain`, `/sensitivity`, `/predict/baseline`; rate limiting, structured error handling, model versioning, no crash-on-missing-model.
- **Frontend:** `frontend/` -- the original `index.html`/`style.css` extended (not replaced) with a fixed DOM bug, a configurable API base, uncertainty/interval display, Shapley-based explanation display, safe habit feedback, a local-only trend view (localStorage, exportable/deletable, no age/country/gender stored), and a method/limits tab. Pure logic factored into `frontend/lib.js` and unit-tested in Node (`frontend/lib.test.js`, 7 tests, all passing).
- **Reproducibility:** `scripts/01`-`12`, each independently re-runnable; `scripts/run_all.py` runs the full pipeline; every number in every document below is generated from `results/`/`models/`, not typed by hand.

## 5. Experiments and Results
Full results: `results/experiment_matrix.csv` (10 experiments with hypothesis, method, measured result, interpretation), `paper/manuscript.md` Section 5, `paper/tables/`, `paper/figures/`. Headline numbers:
- Baseline reproduced exactly: test R2 0.8780, MAE 0.3465 (`results/baseline_reproduction.json`).
- Improved model: test R2 0.8887, MAE 0.3327; improvement CI excludes zero under paired bootstrap (`results/final_vs_baseline.json`).
- Engineered features rejected by pre-registered rule; `Stress_Level` found dispensable, contradicting the project's own initial hypothesis (`results/feature_decision.json`, `results/ablation_results.csv`).
- Locally-weighted conformal intervals meet/exceed nominal coverage at 80/90/95% (`results/uncertainty_evaluation.csv`).
- Subgroup error reported with sample-size caveats; fairness not claimed (`docs/fairness.md`).

## 6. Discussion
`paper/manuscript.md` Section 6.

## 7. Limitations
`paper/manuscript.md` Section 7; `MODEL_CARD.md` "Known limitations and failure modes"; most importantly, **dataset provenance is not established**.

## 8. Ethics
`ETHICS.md`.

## 9. Conclusion
`paper/manuscript.md` Section 9.

## 10. Future Scope
`paper/manuscript.md` Section 10.

## References
`paper/references.bib`.

## Appendices
- Appendix A: Data Quality Report -- `reports/data_quality_report.md`
- Appendix B: Error Analysis -- `reports/error_analysis.md`
- Appendix C: Model Card -- `MODEL_CARD.md`
- Appendix D: Data Card -- `DATA_CARD.md`
- Appendix E: Fairness Report -- `docs/fairness.md`
- Appendix F: Viva Preparation -- `docs/viva_questions.md`
- Appendix G: Deployment Guide -- `docs/DEPLOYMENT.md`
- Appendix H: Prior Audit (superseded) -- `docs/PRIOR_AUDIT_REPORT.md`
