# AUDIT_REPORT.md
**Repository:** https://github.com/Saumyadip2004/Mental-Health-Prediction
**Live deployment (from GitHub "About"):** https://mental-health-prediction-mu.vercel.app
**Audit date:** 2026-09-24
**Method:** Files read directly from GitHub (`main.py`, `index.html`, `requirements.txt`, `README.md`, repo file tree). `ML_Project.ipynb`, `script.js`, `style.css`, and the CSV dataset could **not** be fetched as raw text in this session (GitHub renders them via JavaScript, and this tool cannot execute `git clone` — no network access in the sandbox). Every claim below is sourced from an actual file; anything about the notebook, dataset or frontend JS is explicitly marked **UNKNOWN** rather than guessed, per your Rule #3/#4.

---

## 1. Current Architecture (verified)

```
Student Social Media And Mental Health Impact.csv
        ↓
ML_Project.ipynb   [UNKNOWN CONTENTS — not readable in this session]
        ↓
Mental_Health_Model.pkl   (joblib-serialized object, likely a full sklearn Pipeline
                            or a bare estimator — UNKNOWN which)
        ↓
main.py  →  FastAPI app, single file, no package structure
        ↓
index.html + style.css + script.js  →  static frontend, served separately
        ↓
Deployment: Vercel (frontend confirmed by URL; backend host UNKNOWN —
            Vercel does not natively host long-running FastAPI/uvicorn processes,
            so the backend is almost certainly hosted elsewhere, e.g. Render/Railway.
            VERIFY: check script.js for the actual API base URL it calls.)
```

## 2. Current Files (from repo tree)

| File | Purpose | Status |
|---|---|---|
| `ML_Project.ipynb` | Training notebook, 3394 lines / 494 KB | Not readable this session |
| `Mental_Health_Model.pkl` | Serialized trained model | Exists; internals unknown |
| `Student Social Media And Mental Health Impact.csv` | Dataset | Not readable this session |
| `main.py` | FastAPI backend | **Read in full** (79 lines) |
| `index.html` | Frontend markup | **Read in full** (236 lines) |
| `script.js` | Frontend fetch/validation logic | Not readable this session |
| `style.css` | Styling | Not readable this session |
| `requirements.txt` | Backend deps | **Read in full** |
| `README.md` | Just the repo title, 2 lines | **Read in full** |
| `__pycache__/` | Compiled bytecode | **Committed to git — should not be** |

No `.gitignore`, no `Dockerfile`, no `tests/`, no `.env.example`, no CI config, no `LICENSE` visible in the tree.

## 3. Current ML Pipeline — what's verifiable from `main.py` only

```python
model = joblib.load('Mental_Health_Model.pkl')
...
prediction = model.predict(input_row)[0]
```

The API sends `model.predict()` a **single-row pandas DataFrame** with these exact columns:
`Age, Gender, Country, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Avg_Daily_Usage_Hours, Daily_Unlocks, Study_Hours, Physical_Activity_Hours, Sleep_Hours_Per_Night, Stress_Level, Grouped_country`

**Notable finding:** the row contains **both** `Country` (raw, unbounded cardinality — could be any string the user types, since the frontend `<input>` is a free-text field with a datalist, not a strict dropdown) **and** `Grouped_country` (bucketed into `top_countries` + `"Other"`). Two possibilities:
- The trained pipeline only selects `Grouped_country` via a `ColumnTransformer`, and `Country` is harmlessly ignored → fine, just dead code sitting in the request payload.
- The trained pipeline actually encodes raw `Country` too (e.g. via a `OneHotEncoder` fit on the training set's country values) → then any country the pipeline didn't see during training will either error or silently degrade, because `country` is free-text on the frontend but presumably came from a fixed category list during training.

**This can't be resolved without the notebook.** ACTION FOR YOU: open the notebook and check literally which columns the `ColumnTransformer`/`Pipeline` selects. This is exactly the kind of silent bug Phase 0 exists to catch — I'm not going to guess an answer here.

Everything else Phase 0.3 asks for — train/test split strategy, whether scaling/encoding happens inside or outside a `Pipeline`, cross-validation, random seed, hyperparameters, which of the 10 candidate model types was chosen, evaluation metrics obtained — is **UNKNOWN**, because it lives only in `ML_Project.ipynb`, which I could not read.

**How to verify (your action):** paste the notebook's cells here (especially the imports, the `train_test_split` call, the `ColumnTransformer`/`Pipeline` definition, the model `.fit()` call, and any printed metrics), or upload the `.ipynb` file directly to this chat so I can read it with the file tools. Once I have it I'll do the full 0.3 leakage/methodology audit properly.

## 4. Data audit — status: blocked, needs your action

I could not open the CSV (GitHub only renders large CSVs through a JS-driven table UI that this fetch tool can't execute). To do Phase 0.2 properly (rows, dtypes, nulls, duplicates, target distribution, category cardinality, leakage risk, demographic representation) I need the actual data. **Please upload the CSV to this chat** — once it's in `/mnt/user-data/uploads/` I'll run a full reproducible profiling pass with pandas and give you real numbers, not estimates.

## 5. API Audit (`main.py`, verified)

**Good:**
- Pydantic `BaseModel` with `Literal` types and `Field(ge=..., le=...)` bounds on every categorical/numeric input — this is genuinely solid input validation for a student project.
- `response_model=PredictionResponse` enforces output typing.
- Country bucketing (`top_countries` fallback to `"Other"`) shows awareness of high-cardinality categorical handling.

**Problems found (concrete, not speculative):**
1. **CORS wide open** — `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`. Fine for a local demo, unacceptable once deployed with a real domain. (Phase 17.)
2. **No `/health` endpoint** — Phase 15 asks for one; currently absent.
3. **No `/model-info` endpoint**, no `model_version` field anywhere — you can't tell which model produced a given prediction.
4. **No structured error handling** — a malformed request just gets FastAPI's default 422; a model-loading failure at import time would crash the whole app with no graceful fallback.
5. **No logging at all.**
6. **`GET /` returns a set literal** (`{'Welcome to Saumyadip Das World Guys'}`) rather than a dict — this actually works because FastAPI can serialize a set, but it's a code smell; sets are unordered and this isn't intentional API design.
7. **Model loaded at module import time with no try/except** — if `Mental_Health_Model.pkl` is missing, corrupted, or was pickled with a different sklearn version than `requirements.txt` pins (`scikit-learn==1.9.0`), the app fails to boot with an unhelpful trace.
8. **No rate limiting, no request size limits** — low priority for a student project but worth naming in `ETHICS.md`/`MODEL_CARD.md` as a known limitation given the mental-health framing.

## 6. Frontend Audit (`index.html`, verified; `script.js`/`style.css` unread)

**Good:**
- Semantic `<fieldset>`/`<legend>` grouping, proper `<label for>` associations, ARIA (`aria-live="polite"` on the result panel, `aria-hidden` on decorative SVG).
- Explicit loading state (`state-loading`), error state (`state-error`), and idle state markup already exist — the skeleton for good UX is there.
- A visible disclaimer in the footer: *"Built for informational purposes only — this is not a clinical assessment."* — this is exactly the kind of language Phase 26 (Ethics) wants, and it's already present. Good sign the original author had some awareness of this.
- Country field is free text with a `<datalist>` suggestion list rather than a rigid dropdown — nice UX, but see the pipeline risk noted in §3 above: this is precisely what could feed the model a `Country` value it never saw in training.

**Can't audit without `script.js`:** actual fetch logic, error message content, client-side validation logic, retry behavior, how the gauge/score visualization maps a float to the arc — all **UNKNOWN**. Please paste `script.js` and `style.css` content directly into the chat (they're small enough — the file tree shows no huge size flags) and I'll complete Phase 0.5 properly.

## 7. Risks (ranked)

1. **Notebook/data unaudited** — biggest blocker. Can't assess leakage, split strategy, or even confirm what "Mental_Health_Score" range/distribution actually looks like. Everything downstream (Phases 1–30) depends on this.
2. **Country/Grouped_country duplication** — possible silent train/serve skew.
3. **No preprocessing-inside-pipeline guarantee confirmed** — the `.pkl` might be a bare estimator expecting pre-encoded input, in which case `main.py`'s raw-string DataFrame would just crash at request time unless it's actually a full `Pipeline`. Given `predict()` is called directly on a DataFrame of raw strings and it apparently works in production, it's *likely* a full `Pipeline` object — but "likely" isn't "confirmed."
4. **No versioning anywhere** — model file, API, frontend are all unversioned. A future retrain silently changes behavior with no audit trail.
5. **Mental-health framing with zero ethics/disclaimer infrastructure in the backend** (the disclaimer exists only in the HTML footer, not enforced anywhere in the API contract or a `MODEL_CARD.md`).

## 8. Research Opportunities (tentative — pending RQ selection with you, Phase 1)

Based on the feature set actually confirmed in `main.py` (age, gender, country, academic level, platform, purpose, usage hours, unlocks, study/activity/sleep hours, stress level → continuous score), RQ1 (predictive performance), RQ2 (feature attribution via SHAP), and RQ4 (subgroup performance by gender/academic level/country-group) all look directly answerable with this schema. RQ5 (uncertainty) and RQ6 (longitudinal) are architecturally clean additions since the current API is stateless — nothing to migrate away from.

## 9. Proposed Final Architecture

Not proposing model/feature changes yet — Rule #6/#7 says don't rewrite unnecessarily and every change needs a reason. Once I've seen the notebook and dataset, Stage 2 (Phase 2, clean baseline) will either confirm the existing pipeline is sound (in which case we wrap it in tests + docs and move to explainability) or identify a leakage/methodology problem that justifies a retrain.

## 10. Proposed Research Questions

Holding RQ1, RQ2, RQ4, RQ5 as strong candidates per §8. Will finalize in Phase 1 once data audit confirms sample size is sufficient for subgroup analysis (RQ4/fairness needs enough rows per subgroup to be statistically meaningful — currently unknown).

## 11. Prioritized Roadmap (next 3 actions, blocking everything else)

1. **You upload:** the CSV, the `.ipynb` (or paste its key cells), `script.js`, `style.css`.
2. **I run:** full Phase 0.2 data-quality report + Phase 0.3 ML/leakage audit with actual pandas output, not estimates.
3. **Then:** Phase 1 (lock the research questions) → Phase 2 (clean baseline pipeline with proper CV).

## 12. What should NOT be changed (yet)

- `main.py`'s Pydantic schema — it's well-built; don't touch until the notebook confirms the model's real expected input.
- The disclaimer language in `index.html`'s footer — keep it, extend it, don't remove it.
- `requirements.txt` version pins — no reason yet to bump anything.

## What files need to be created (once data is available)

- `DATA_CARD.md`, `MODEL_CARD.md`, `ETHICS.md` (Phases 24–26)
- `results/`, `figures/`, `docs/` directories
- `.gitignore` (to stop committing `__pycache__/`)
- `backend/tests/` (currently zero tests exist)

---
**Bottom line:** I've audited everything that's actually readable from the repo (`main.py`, `index.html`, `requirements.txt`, file tree) and found it in better shape than the median "tutorial project" — decent Pydantic validation, an existing disclaimer, sensible-looking API contract. But I genuinely cannot audit the ML methodology, the dataset, or the frontend JS without you uploading them directly, and I'm not going to fabricate that section. Upload the CSV + notebook + script.js/style.css and I'll finish Phase 0 for real.
