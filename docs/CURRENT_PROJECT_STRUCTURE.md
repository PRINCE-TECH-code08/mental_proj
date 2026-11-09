# Repository structure

```
baseline/                 ORIGINAL project, byte-identical (CHECKSUMS.sha256; CI verifies)
  main.py, ML_Project.ipynb, Mental_Health_Model.pkl, index.html, script.js, style.css, requirements.txt, README.md
data/raw/                 original CSV (read-only) + checksum
src/mhp/                  config, data, features (leak-free Pipeline), models, metrics, explain (MC-Shapley), service (all API logic)
scripts/                  01..11 reproducible pipeline (run_all.py runs 01-10; 11 builds docs/tables)
results/  figures/  reports/   generated outputs
models/                   model_bundle.joblib, metadata.json, conformal.json, reference.json, noise_floor.json, background.csv
backend/app.py            FastAPI v2 (backward-compatible /predict)
frontend/                 improved static site (config.js, lib.js + tests)
tests/                    unittest/pytest tests (data, ML, service, API)
docs/  paper/  MODEL_CARD.md  DATA_CARD.md  ETHICS.md
Dockerfile  docker-compose.yml  render.yaml  .github/workflows/ci.yml
```
Original repository layout (from the prior audit): flat root with `main.py`, `index.html`, `script.js`, `style.css`, notebook, pkl, csv, `requirements.txt`, 2-line README, committed `__pycache__/`.
