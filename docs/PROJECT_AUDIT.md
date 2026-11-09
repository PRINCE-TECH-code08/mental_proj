# Project Audit (Phase 0) - what was found, what was done

Method: every uploaded file was read and executed (dataset profiled, notebook cells read, pickle loaded and re-scored, frontend JS checked against the HTML). The earlier `PRIOR_AUDIT_REPORT.md` could not read the notebook/CSV; its open questions are answered here.

## Current architecture (as it actually was)
`CSV -> notebook (clean, group country, 70/30 split, ColumnTransformer+Pipeline, LR/RF/tuned RF) -> Mental_Health_Model.pkl (the DEFAULT RandomForest, saved from rf_pipeline) -> main.py (FastAPI /predict) -> static HTML/CSS/JS`. Backend host: not in repo (a Render URL appears only in a comment in `script.js`).

## Answers to the prior audit's UNKNOWNs
| Question | Answer (verified) |
|---|---|
| Is the pickle a full Pipeline? | Yes: `Pipeline(preprocessor, RandomForestRegressor(random_state=42))`, 12 input features. |
| Does `Country` matter (train/serve skew)? | No. The model's `feature_names_in_` contains `Grouped_country` but not `Country`; the raw `Country` sent by `main.py` is ignored. Hard-coded `top_countries` in `main.py` equals the notebook's list (verified). |
| Baseline metrics? | Reproduced exactly: test R2 0.878017, MAE 0.346502 (n=1500); train R2 0.980907 (`results/baseline_reproduction.json`). |
| Which model is saved? | The untuned RF. The notebook's tuned RF had lower test R2 (0.8652), and the untuned one was chosen after viewing test scores (selection on the test set). |
| Split / leakage? | Split before preprocessing fit (good). Country top-10 list was computed on the full frame before the split (mild, unsupervised; same list results when computed on train only). |

## Priority matrix
| # | Issue | Severity | Research impact | Engineering impact | Action / status |
|---|---|---|---|---|---|
| 1 | `script.js` reads `#error-label`, which does not exist in `index.html` -> any error throws, UI stays on "loading" | High | - | High | **Fixed** (element added; DOM-contract check) |
| 2 | Pickle saved with scikit-learn 1.9.1 but `requirements.txt` pins 1.9.0 | High | Reproducibility | High | **Fixed**: bundle trained/tested with pinned 1.8.0; metadata records versions |
| 3 | API accepts age 10-100; data covers 18-24 (also other ranges) -> silent extrapolation | High | Validity | Medium | **Fixed**: out-of-domain warnings; interval flagged unreliable |
| 4 | Frontend `API_BASE` hard-coded to `127.0.0.1:8000`; error text says port 2200 | High | - | High | **Fixed**: `config.js` |
| 5 | Model selected on test set; tuning judged on same test | Medium | Validity | - | **Fixed**: nested CV; test used once |
| 6 | Unbounded RF, train R2 0.98 vs test 0.88 | Medium | Generalisation | - | Documented; tuned RF (nested CV) modest gain |
| 7 | CORS `*` | Medium | - | Security | **Fixed** (env allow-list) |
| 8 | UI bands (<4 strained, <7 balanced) arbitrary and unvalidated; observed score range 3.6-9.4 | Medium | Interpretation | - | **Replaced** by percentile position vs reference data |
| 9 | Negative `Physical_Activity_Hours` (10 rows) silently clipped; 2 duplicate rows dropped | Low | Data quality | - | Kept for comparability; logged |
| 10 | `GET /` returns a Python set | Low | - | Low | **Fixed** |
| 11 | No tests, health check, logging, versioning, rate limit | Medium | - | High | **Fixed** |
| 12 | Stray `S` after `</head>` | Low | - | Low | **Fixed** |
| 13 | `__pycache__` committed, no `.gitignore`, empty README | Low | - | Low | **Fixed** in new layout (run `git rm -r --cached __pycache__` in the old repo) |
| 14 | `Stress_Level` conceptually near the outcome | Medium | Interpretation | - | Ablation: removing it costs only 0.0025 MAE; documented |
| 15 | Dataset provenance undocumented; description suggests designed/simulated data | **High** | **Generalisability** | - | Cannot be resolved from files; stated as a limitation everywhere |

## What cannot be claimed
Causation (screen time -> wellbeing), clinical validity, generalisation beyond this file, fairness, or that changes in predictions over time are changes in a person's mental health.
