# Student Digital Wellbeing: Explainable, Uncertainty-Aware, Fairness-Evaluated Prediction

Final-year research project. Starts from a preserved copy of an existing student mental-health prediction app (`baseline/`, byte-identical to the original, checksum-verified in CI) and builds a reproducible research pipeline and improved full-stack system around it.

**Not a diagnostic tool.** Predicts a dataset-specific self-reported wellbeing score from habits; it does not diagnose or treat any condition. See `ETHICS.md`.

## What changed vs the baseline (measured, not asserted)
| | Baseline (original pickle) | Improved |
|---|---|---|
| Test MAE | 0.3465 | **0.3327** |
| Test R2 | 0.8780 | **0.8887** |
| Model selection | picked by looking at test-set scores | nested CV on training data only |
| Uncertainty | none | cross-conformal interval, coverage-checked |
| Explanation | none | Monte-Carlo Shapley (efficiency + convergence checked) |
| Subgroup reporting | none | per-group MAE/R2/bias with sample-size flags; fairness NOT claimed |
| API | CORS `*`, no health check, no logging, model load could crash the app | `/health`, `/model-info`, rate limiting, structured errors, versioned model, backward-compatible `/predict` |
| Frontend bug | `script.js` read a DOM id (`error-label`) that didn't exist in `index.html`, so errors silently failed | fixed; DOM-contract checked by a script; trend view, CSV export, local-only history added |
| Tests | none | 23 unit/integration tests (`tests/`), `frontend/lib.test.js` |

Full numbers: `results/final_vs_baseline.json`, `MODEL_CARD.md`, `paper/manuscript.md`.

## Research questions answered
1. Does a leak-free, nested-CV-tuned model beat the original? Yes, modestly (`results/final_vs_baseline.json`).
2. Which feature groups drive accuracy? Demographics+social-media alone nearly matches all features; `Stress_Level` is dispensable (`results/ablation_results.csv`) -- this *contradicts* our own pre-registered hypothesis.
3. Do engineered ratio features help? No (`results/feature_decision.json`).
4. Is the model's uncertainty well-calibrated? Locally-weighted conformal intervals meet or exceed nominal coverage at 80/90/95% (`results/uncertainty_evaluation.csv`).
5. Does error vary by subgroup? Yes, unevenly and with small-sample groups flagged; **fairness is not claimed** (`docs/fairness.md`).
6. How much does the prediction move from input noise alone? ~0.55 points (95th percentile) for 0.25h of assumed self-report noise (`results/sensitivity_extra.json`) -- used by the frontend to caveat trend changes.

## Repository layout
See `docs/CURRENT_PROJECT_STRUCTURE.md`.

## Quickstart
```bash
pip install -r requirements-dev.txt
python scripts/run_all.py            # re-run the full research pipeline (~15 min, 1 CPU)
python -m unittest discover -s tests -v
uvicorn app:app --app-dir backend --reload --port 8000
python -m http.server 8080 --directory frontend   # then set frontend/config.js to point at your API
```
Deployment (Render + Vercel, or Docker): `docs/DEPLOYMENT.md`.

## Key documents
- `docs/PROJECT_AUDIT.md` -- Phase 0 audit, priority matrix, what was fixed
- `reports/data_quality_report.md`, `reports/error_analysis.md`
- `MODEL_CARD.md`, `DATA_CARD.md`, `ETHICS.md`, `docs/fairness.md`
- `docs/literature_review.md`, `docs/research_gap.md` (partial; no novelty claimed)
- `results/experiment_matrix.csv` -- every experiment, hypothesis, method, measured result, interpretation
- `paper/manuscript.md`, `paper/references.bib`, `paper/tables/`, `paper/figures/`
- `docs/viva_questions.md`, `docs/FINAL_PROJECT_REPORT.md`, `docs/FINAL_QUALITY_AUDIT.md`

## Known limitations (see `MODEL_CARD.md` for the full list)
Dataset provenance is not established (source describes the 5,000 rows as "designed" for educational use, not a documented survey); ages 18-24 and two gender categories only; `Country` behaves as a demographic proxy with non-trivial importance; conditional interval coverage and subgroup errors vary by group; explanations and sensitivity results describe model behaviour, not causal effects.

## Environment note
This project was built in a sandbox with no outbound network access, so `pip install shap`, `docker build`, live deployment, and the FastAPI `TestClient` tests could not be executed there. Everything network-independent (data pipeline, model training/evaluation, explainability, uncertainty, fairness, sensitivity, unit tests, the pure frontend-logic tests) was executed and verified. `.github/workflows/ci.yml` runs the remaining tests (API, Docker build) on GitHub's runners, which do have network access.

## Citation
If you use this repository, please cite `paper/manuscript.md` and see `paper/references.bib` for the literature it builds on.
