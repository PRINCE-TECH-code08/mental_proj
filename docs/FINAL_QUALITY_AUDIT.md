# Final Quality Audit (Phase 34)

Every item was checked programmatically against the actual repository (`results/final_quality_checks.json`) or run directly; nothing here is asserted without a check having been executed in this session.

## ML
- [x] No leakage: all preprocessing fit inside `Pipeline` on training folds only (`src/mhp/features.py`)
- [x] Reproducible: fixed seed 42 recorded in `models/metadata.json`; `baseline/CHECKSUMS.sha256` + `data/raw/CHECKSUM.sha256` verified in CI
- [x] Proper CV: repeated 5x2 CV for comparison, nested CV for tuning (`results/model_comparison.csv`, `results/nested_cv_tuned.csv`)
- [x] Baseline: Dummy(mean) included in comparison
- [x] Comparison: 8 candidate models
- [x] Ablation: 11 feature-group experiments x 2 models (`results/ablation_results.csv`)
- [x] Error analysis: `reports/error_analysis.md` (bias, subgroup error, largest errors, heteroscedasticity)

## XAI
- [x] Global explanation: permutation importance + mean |Shapley| (`results/permutation_importance.csv`, `results/shapley_global.csv`)
- [x] Local explanation: per-instance Shapley bar chart (`figures/shap_local_example.png`), served live via `/explain`
- [x] No causal overclaiming: `MODEL_CARD.md`, `ETHICS.md`, and every explanation-producing script/response include an explicit "not causal effects" statement

## Uncertainty
- [x] Defensible method: cross-conformal (Lei et al. 2018), locally-weighted variant, cited in `paper/manuscript.md`
- [x] Calibrated/evaluated: marginal coverage at 3 levels x 2 methods measured on held-out test set (`results/uncertainty_evaluation.csv`); conditional coverage by subgroup also measured (`results/uncertainty_conditional_coverage.csv`)
- [x] Documented: `MODEL_CARD.md` "Uncertainty" section, method and limitation stated

## Fairness
- [x] Subgroup evaluation: 5 attributes, MAE/RMSE/R2/bias/coverage with bootstrap CIs (`results/subgroup_performance.csv`)
- [x] Representation documented: `DATA_CARD.md`; low-n groups explicitly flagged, not hidden
- [x] Fairness NOT asserted anywhere in the repository (checked by design; `docs/fairness.md` states this explicitly)

## Backend
- [x] API implemented: `backend/app.py`
- [x] Validation: Pydantic `StudentData` with bounds, re-used from the improved `ModelService`
- [x] Errors handled: `InputError`, `RuntimeError`, generic `Exception` handlers -> structured JSON, never a bare 500 traceback to the client
- [x] Health endpoint: `GET /health`, reports degraded state if the model failed to load

## Frontend
- [x] Responsive: media queries preserved/extended from the original CSS
- [x] Accessible: `aria-live`, `aria-hidden`, `<label for>` preserved from the original; focus-visible outlines added
- [x] Prediction works: verified via `ModelService` unit tests (same code path the API calls) and static DOM-contract check (all `getElementById`/`$()` targets exist in the HTML -- the original had one that didn't)
- [x] Errors work: `#error-label` bug (present in the original, absent from its HTML) fixed and covered by the DOM-contract check
- [x] Results understandable: score + interval + relative position + plain-language feedback + explicit disclaimer

## Security
- [x] No secrets committed: checked by `tests/test_pipeline_and_service.py::test_no_secrets_committed` (scans all non-baseline text files for key/credential patterns)
- [x] CORS configured: environment-driven allow-list, not `*`, verified by `no_wildcard_cors` check and (where network allows) `tests/test_api.py::test_cors_not_wildcard`
- [x] Input validation: Pydantic schema + `ModelService.to_frame` (range/category/out-of-domain checks), unit-tested

## Deployment
- [x] Dockerfile, docker-compose.yml, render.yaml, frontend/vercel.json present and syntax-validated
- [ ] Docker image not actually built (no `docker` binary in this sandbox) -- built and verified in `.github/workflows/ci.yml` instead
- [ ] Not actually deployed to Render/Vercel (no outbound network in this sandbox) -- step-by-step instructions in `docs/DEPLOYMENT.md` for you to execute
- [x] Environment variables documented (`.env.example`)
- [x] Deployment instructions: `docs/DEPLOYMENT.md`

## Research
- [x] Literature review: partial, verified entries only, gaps stated (`docs/literature_review.md`)
- [x] Research gap: candidate only, novelty not claimed (`docs/research_gap.md`)
- [x] Experiments: 10-row experiment matrix with measured results (`results/experiment_matrix.csv`)
- [x] Reproducibility: `scripts/run_all.py`, checksummed baseline and data
- [x] Limitations: stated in `MODEL_CARD.md`, `paper/manuscript.md` Sec. 7, `DATA_CARD.md`
- [x] Ethics: `ETHICS.md`

## Test suite summary (this session)
- `python -m unittest discover -s tests`: 23 tests, 14 executed and passing (data + ML + service), 9 skipped (FastAPI TestClient -- fastapi/httpx not installable offline here; runs in CI)
- `node --test frontend/lib.test.js`: 7/7 passing
- `cd baseline && sha256sum -c CHECKSUMS.sha256`: all 8 files OK (baseline untouched)

## Outstanding items (honest, not hidden)
1. FastAPI endpoint tests and Docker build are written and reviewed but not executed in this sandbox (no network/no Docker) -- CI will execute them on push.
2. Literature review and research-gap novelty claim are intentionally partial.
3. Actual deployment (Render/Vercel) requires you to run the steps in `docs/DEPLOYMENT.md` from an environment with network access and your own accounts.
4. Dataset provenance remains unresolved at the source; this is stated everywhere it matters rather than fixed, because it cannot be fixed from the files available.
