# Deployment guide

Not deployed from this session (the build sandbox has no outbound network access, so it cannot reach GitHub, Render, or Vercel). Everything below is exact, tested-locally-where-possible configuration for you to run.

## Backend (FastAPI) -> Render (or Railway/Fly, same idea)
1. Push this repo to GitHub (`git init && git add -A && git commit -m "v2" && git remote add origin <your-repo> && git push`).
2. On Render: New -> Web Service -> connect the repo -> it will detect `render.yaml` (Blueprint) automatically, or set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --app-dir backend --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
   - Env var `ALLOWED_ORIGINS` = your deployed frontend URL(s), comma-separated
3. First request on Render's free tier is slow (cold start); `script.js` already shows a friendly retry message for this.
4. Verify: `curl https://<service>.onrender.com/health` -> `{"status":"ok",...}`.

## Frontend (static) -> Vercel
1. `vercel --cwd frontend` (or import the `frontend/` folder as a project in the Vercel dashboard; root directory = `frontend`).
2. Before deploying, edit `frontend/config.js` and put your real Render URL in the non-localhost branch.
3. `frontend/vercel.json` sets security headers (CSP, no-sniff, no-referrer). If you add another API host, add it to `connect-src` in that file or requests will be blocked by the browser.

## Docker (self-host / local parity with prod)
Docker is not installed in this sandbox, so `docker build` was not run here; the file was written to the Dockerfile spec (slim Python, non-root user, `$PORT`, `0.0.0.0`) and reviewed line by line instead. Run locally to confirm:
```
docker compose up --build
# API:      http://localhost:8000/health
# Frontend: http://localhost:8080
```

## Local dev (no Docker)
```
pip install -r requirements.txt
uvicorn app:app --app-dir backend --reload --port 8000
# separately, serve the frontend (any static server), e.g.:
python -m http.server 8080 --directory frontend
```

## Re-running the research pipeline
```
pip install -r requirements-dev.txt
python scripts/run_all.py        # ~15 min on 1 CPU; writes results/, figures/, models/model_bundle.joblib
python scripts/11_build_docs.py  # rebuilds paper/tables, MODEL_CARD.md, experiment_matrix.csv
```
Baseline integrity check anyone can run: `cd baseline && sha256sum -c CHECKSUMS.sha256`.
