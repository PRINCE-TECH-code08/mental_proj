# PROJECT PROGRESS (living document — update at the end of every session)

**Repo root:** `/home/claude/Mental-Health-Prediction` (not yet pushed to git/GitHub by me — no network in this sandbox)
**Last updated:** this session, after Phase 34.

## STATUS: Core roadmap complete. Remaining items require network access / your action (listed below).

## Completed (verified, not just claimed)
- [x] Phase 0: Audit — `docs/PROJECT_AUDIT.md`, `docs/PRIOR_AUDIT_REPORT.md`, `docs/CURRENT_PROJECT_STRUCTURE.md`
- [x] Baseline preserved byte-identical — `baseline/` + `CHECKSUMS.sha256` (verified: all OK)
- [x] Phase 1: Data audit — `reports/data_quality_report.md`, `results/cleaning_log.json`
- [x] Phase 2: Leak-free pipeline + model comparison — `src/mhp/features.py`, `results/model_comparison.csv`
- [x] Phase 3: Hyperparameter tuning (nested CV) — `results/nested_cv_tuned.csv`, `results/hyperparameter_search.csv`, `results/selected_model.json`
- [x] Phase 4: Feature engineering — `results/engineered_features.csv`, `results/feature_decision.json` (rejected)
- [x] Phase 5: Ablation study — `results/ablation_results.csv`, `results/ablation_which_demographic.csv`, `figures/ablation_comparison.png`
- [x] Final model trained + evaluated vs baseline (paired bootstrap) — `results/final_vs_baseline.json`, `models/model_bundle.joblib`
- [x] Phase 6: Error analysis — `reports/error_analysis.md`
- [x] Phase 7: Explainability (Monte-Carlo Shapley, in-house — `shap` unavailable offline; validated for efficiency + convergence) — `src/mhp/explain.py`, `results/shapley_*`
- [x] Phase 8: Uncertainty (cross-conformal, locally-weighted) — `models/conformal.json`, `results/uncertainty_evaluation.csv`, `results/uncertainty_conditional_coverage.csv`
- [x] Phase 9: Fairness/subgroup — `results/subgroup_performance.csv`, `docs/fairness.md` (fairness NOT claimed)
- [x] Phase 10/11: Sensitivity + noise floor (for longitudinal trend interpretation) — `results/sensitivity_summary.csv`, `results/sensitivity_extra.json`, `models/noise_floor.json`
- [x] Phase 12: Personalized feedback (safe, no diagnosis language) — `src/mhp/service.py::feedback`
- [x] Phase 13: Questionnaire module — evaluated, deliberately NOT integrated (no license/methodology basis); not fabricated
- [x] Phase 14: Frontend — original HTML/CSS extended (not replaced); DOM-bug fixed; trend tab (localStorage), method/limits tab; `frontend/lib.js` unit-tested (7/7 Node tests)
- [x] Phase 15: FastAPI v2 — `backend/app.py` (backward-compatible `/predict`, + `/health`, `/model-info`, `/explain`, `/predict-with-uncertainty`, `/sensitivity`, `/predict/baseline`)
- [x] Phase 16: Model versioning — `models/metadata.json`
- [x] Phase 17: Database — evaluated, deliberately NOT added (stateless API + client-side history is sufficient and more private); documented decision
- [x] Phase 18: Security — `.env.example`, `.gitignore`, CORS allow-list, rate limiting, no secrets (tested)
- [x] Phase 19: Testing — `tests/test_pipeline_and_service.py` (14 tests, passing), `tests/test_api.py` (9 tests, written+reviewed, SKIPPED here — no network to install fastapi/httpx; will run in CI), `frontend/lib.test.js` (7 tests, passing)
- [x] Phase 20: Docker — `Dockerfile`, `docker-compose.yml`, `.dockerignore` (NOT built here — no docker binary; builds in CI)
- [x] Phase 21: CI — `.github/workflows/ci.yml` (lint, baseline-checksum, unit tests, node tests, baseline reproduction, docker build)
- [x] Phase 22: Deployment config — `render.yaml`, `frontend/vercel.json`, `docs/DEPLOYMENT.md` (NOT actually deployed — no network; instructions are exact and ready for you to run)
- [x] Phase 23/24: Literature review + research gap — `docs/literature_review.md`, `docs/research_gap.md` (PARTIAL, verified entries only, no novelty claimed)
- [x] Phase 25: Experiment matrix — `results/experiment_matrix.csv` (10 experiments, measured)
- [x] Phase 26: Figures — 14 figures in `figures/` + copied to `paper/figures/`
- [x] Phase 27: Model card — `MODEL_CARD.md`
- [x] Phase 28: Data card — `DATA_CARD.md`
- [x] Phase 29: Ethics — `ETHICS.md`
- [x] Phase 30: Research paper — `paper/manuscript.md`, `paper/references.bib`, `paper/supplementary_material.md`, `paper/tables/`, `paper/figures/`
- [x] Phase 31: GitHub README — `README.md`
- [x] Phase 32: Viva prep — `docs/viva_questions.md`
- [x] Phase 33: Final project report — `docs/FINAL_PROJECT_REPORT.md`
- [x] Phase 34: Final quality audit — `docs/FINAL_QUALITY_AUDIT.md`, `results/final_quality_checks.json` (all programmatic checks PASS)

## Files changed/created this continuation session
`README.md`, `docs/DEPLOYMENT.md`, `tests/test_api.py` (appended sandbox note), `paper/manuscript.md`, `paper/references.bib`, `paper/supplementary_material.md`, `scripts/12_build_paper.py`, `docs/viva_questions.md`, `docs/FINAL_PROJECT_REPORT.md`, `docs/FINAL_QUALITY_AUDIT.md`, `results/final_quality_checks.json`, `PROJECT_PROGRESS.md` (this file).
(Everything from the earlier session — scripts 01-11, src/mhp/*, backend/app.py, frontend/*, tests/test_pipeline_and_service.py, MODEL_CARD.md, DATA_CARD.md, ETHICS.md, docs/PROJECT_AUDIT.md etc. — was preserved untouched; re-verified, not re-run, except where noted.)

## Known issues / honest gaps (see docs/FINAL_QUALITY_AUDIT.md "Outstanding items")
1. `tests/test_api.py` (FastAPI TestClient, 9 tests) not executed here — no network to `pip install fastapi httpx uvicorn`. Will run in `.github/workflows/ci.yml` on push.
2. Docker image not built here — no `docker` binary. Builds in CI.
3. Not deployed to Render/Vercel — no outbound network in this sandbox. `docs/DEPLOYMENT.md` has exact steps for you.
4. Literature review / research gap are intentionally partial (no systematic search completed); no novelty claim made.
5. Dataset provenance unresolved at the source — documented everywhere as a limitation, not fixable from the files available.

## EXACT NEXT TASK (if a future session resumes this)
All roadmap phases are complete on this end. The next task is **environment-dependent, not code-dependent**:
1. `git init && git add -A && git commit -m "v2: research pipeline + improved system" && git push` to your GitHub.
2. Confirm `.github/workflows/ci.yml` goes green (it will actually execute `tests/test_api.py` and the Docker build for the first time).
3. Follow `docs/DEPLOYMENT.md` to deploy the backend (Render) and frontend (Vercel); update `frontend/config.js`'s production `API_BASE` to the real Render URL.
4. Fill in the placeholders in `docs/FINAL_PROJECT_REPORT.md` (institution, names, dates) and `paper/manuscript.md` if your department requires author names in the paper itself.
5. Optional: if you get network access to `pip install shap`, run a comparison of the in-house Monte-Carlo Shapley values against real SHAP as a cross-check (flagged as future work in the paper).

If resuming: re-run `(cd baseline && sha256sum -c CHECKSUMS.sha256)` and `python -m unittest discover -s tests` first to confirm nothing regressed, then proceed with the numbered list above.
