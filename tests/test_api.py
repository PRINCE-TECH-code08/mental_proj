"""API tests (FastAPI TestClient). Skipped automatically if fastapi/httpx are not installed. CI installs them."""
import os, sys, unittest, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
try:
    from fastapi.testclient import TestClient
    os.environ["ENABLE_BASELINE"] = "1"; os.environ["RATE_LIMIT_PER_MIN"] = "0"
    import app as appmod
    HAVE = True
except Exception:
    HAVE = False
GOOD = dict(age=21, gender="Female", country="India", academic_level="Undergraduate", most_used_platform="Instagram", purpose_of_use="Entertainment",
            avg_daily_usage_hours=5.0, daily_unlocks=170, study_hours=3.0, physical_activity_hours=1.7, sleep_hours_per_night=6.6, stress_level="Medium")


@unittest.skipUnless(HAVE, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = TestClient(appmod.app); cls.c = cls.cm.__enter__()
    @classmethod
    def tearDownClass(cls): cls.cm.__exit__(None, None, None)

    def test_root_is_a_dict_not_a_set(self): self.assertIn("service", self.c.get("/").json())
    def test_health(self): self.assertEqual(self.c.get("/health").json()["status"], "ok")
    def test_model_info(self): self.assertEqual(self.c.get("/model-info").json()["n_test"], 1500)
    def test_predict_original_contract(self):
        r = self.c.post("/predict", json=GOOD); self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json()["predicted_mental_health_score"], float)
    def test_uncertainty_and_explain(self):
        self.assertEqual(self.c.post("/predict-with-uncertainty?level=0.95", json=GOOD).json()["interval"]["level"], 0.95)
        self.assertEqual(len(self.c.post("/explain", json=GOOD).json()["explanation"]["contributions"]), 12)
    def test_422_shape_matches_fastapi(self):
        r = self.c.post("/predict", json={**GOOD, "gender": "x"}); self.assertEqual(r.status_code, 422); self.assertEqual(r.json()["detail"][0]["loc"][-1], "gender")
    def test_baseline_endpoint(self): self.assertEqual(self.c.post("/predict/baseline", json=GOOD).status_code, 200)
    def test_cors_not_wildcard(self):
        r = self.c.options("/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        self.assertNotEqual(r.headers.get("access-control-allow-origin"), "*"); self.assertIsNone(r.headers.get("access-control-allow-origin"))
    def test_degraded_when_model_missing(self):
        s = appmod.app.state.service; appmod.app.state.service = None
        try:
            self.assertEqual(self.c.get("/health").status_code, 503); self.assertEqual(self.c.post("/predict", json=GOOD).status_code, 503)
        finally: appmod.app.state.service = s

# NOTE ON THIS TEST FILE'S SKIP STATE:
# In the offline build sandbox used to develop this project, `pip install` has no network access,
# so fastapi/httpx/uvicorn could not be installed and TestAPI above is SKIPPED there (see the printed
# "OK (skipped=9)"). scripts/03-05 etc. do not need these packages and were run and verified normally.
# .github/workflows/ci.yml installs requirements-dev.txt on GitHub's runners (which have internet) and
# runs this file for real; that is where these 9 tests are actually exercised end to end.
