"""FastAPI service v2. Thin wrapper over mhp.service.ModelService (all logic is unit-tested there).
Backward compatible: POST /predict accepts the ORIGINAL request body and still returns `predicted_mental_health_score`."""
import logging, os, sys, time, warnings
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mhp.service import ModelService, InputError

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mhp.api")
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://mental-health-prediction-mu.vercel.app,http://localhost:5500,https://mental-health-api-qrn4.onrender.com,http://127.0.0.1:5500,http://localhost:8080").split(",") if o.strip()]
RATE = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
BUNDLE = Path(os.getenv("MODEL_BUNDLE", ROOT / "models" / "model_bundle.joblib"))
BASELINE = Path(os.getenv("BASELINE_MODEL", ROOT / "baseline" / "Mental_Health_Model.pkl"))
ENABLE_BASELINE = os.getenv("ENABLE_BASELINE", "0") == "1"      # off by default: saves ~25 MB pickle + RAM on small hosts


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app.state.service = ModelService(BUNDLE, BASELINE if ENABLE_BASELINE and BASELINE.exists() else None)
        log.info("model loaded: v%s", app.state.service.meta["model_version"])
    except Exception:                       # do not crash on boot; /health reports degraded and predict returns 503
        log.exception("model failed to load")
        app.state.service = None
    yield


app = FastAPI(title="Student Digital Wellbeing API", version="2.0.0", lifespan=lifespan,
              description="Educational, uncertainty-aware wellbeing-score model. NOT a diagnostic tool.")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    t0 = time.time(); ip = request.client.host if request.client else "?"
    if RATE > 0 and request.method == "POST":
        q = _hits[ip]; now = time.time()
        while q and now - q[0] > 60: q.popleft()
        if len(q) >= RATE:
            return JSONResponse(status_code=429, content={"detail": "Too many requests; please slow down."})
        q.append(now)
    resp = await call_next(request)
    log.info("%s %s %s %.0fms", request.method, request.url.path, resp.status_code, (time.time() - t0) * 1000)   # never log request bodies
    return resp


class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female"]
    country: str = Field(..., min_length=1, max_length=60)
    academic_level: Literal["Undergraduate", "Graduate", "High School"]
    most_used_platform: Literal["Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter", "YouTube", "TikTok", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"]
    purpose_of_use: Literal["Networking", "Education", "Entertainment", "News"]
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0, le=5000)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal["Medium", "Low", "Very High", "High"]


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
    model_version: Optional[str] = None
    interval: Optional[dict[str, Any]] = None
    position: Optional[dict[str, Any]] = None
    input_warnings: list[dict[str, Any]] = []
    feedback: list[str] = []
    explanation: Optional[dict[str, Any]] = None
    disclaimer: Optional[str] = None


def svc(request: Request) -> ModelService:
    s = request.app.state.service
    if s is None:
        raise RuntimeError("model unavailable")
    return s


@app.exception_handler(InputError)
async def input_error(_: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"detail": [{"loc": ["body", e["field"]], "msg": e["message"], "type": "value_error"} for e in exc.errors]})


@app.exception_handler(RuntimeError)
async def unavailable(_: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": "The prediction model is temporarily unavailable."})


@app.exception_handler(Exception)
async def unexpected(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/")
def root():
    return {"service": "Student Digital Wellbeing API", "version": app.version, "docs": "/docs", "health": "/health", "disclaimer": "Educational estimate; not a diagnosis."}


@app.get("/health")
def health(request: Request):
    s = request.app.state.service
    return JSONResponse(status_code=200 if s else 503, content={"status": "ok" if s else "degraded", "model_loaded": s is not None, "model_version": s.meta["model_version"] if s else None})


@app.get("/model-info")
def model_info(request: Request):
    return svc(request).info()


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData, request: Request):
    return svc(request).predict(data.model_dump())


@app.post("/predict-with-uncertainty", response_model=PredictionResponse)
def predict_unc(data: StudentData, request: Request, level: float = Query(0.9, description="0.8, 0.9 or 0.95")):
    return svc(request).predict(data.model_dump(), level=level)


@app.post("/explain", response_model=PredictionResponse)
def explain(data: StudentData, request: Request):
    return svc(request).predict(data.model_dump(), explain=True)


@app.post("/sensitivity")
def sensitivity(data: StudentData, request: Request):
    return svc(request).sensitivity(data.model_dump())


@app.post("/predict/baseline")
def predict_baseline(data: StudentData, request: Request):
    s = svc(request)
    if s.baseline is None:
        return JSONResponse(status_code=404, content={"detail": "Baseline model not enabled (set ENABLE_BASELINE=1)."})
    return s.predict_baseline(data.model_dump())
