"""Framework-free tests (unittest; also collected by pytest). Run: python -m unittest discover -s tests -v"""
import json, sys, unittest, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mhp import config as C, data as D
from mhp.features import FeatureBuilder
from mhp.service import ModelService, InputError

GOOD = dict(age=21, gender="Female", country="India", academic_level="Undergraduate", most_used_platform="Instagram",
            purpose_of_use="Entertainment", avg_daily_usage_hours=5.0, daily_unlocks=170, study_hours=3.0,
            physical_activity_hours=1.7, sleep_hours_per_night=6.6, stress_level="Medium")


class TestData(unittest.TestCase):
    def test_schema_and_cleaning(self):
        raw = D.load_raw(); log = {}; df = D.clean(raw, log)
        self.assertEqual(list(raw.columns), C.RAW_INPUTS + [C.TARGET])
        self.assertEqual(raw.isna().sum().sum(), 0)
        self.assertEqual((log["rows_raw"], log["duplicates_dropped"], log["negative_activity_clipped"], log["rows_clean"]), (5000, 2, 10, 4998))
        self.assertTrue((df.Physical_Activity_Hours >= 0).all())

    def test_split_identical_to_baseline(self):
        Xtr, Xte, ytr, yte = D.split(D.clean(D.load_raw()))
        self.assertEqual((len(Xtr), len(Xte)), (3498, 1500))
        self.assertFalse(set(Xtr.index) & set(Xte.index))

    def test_country_bucketing_learned_from_train_only_and_matches_baseline_set(self):
        Xtr, Xte, _, _ = D.split(D.clean(D.load_raw()))
        fb = FeatureBuilder().fit(Xtr)
        self.assertEqual(set(fb.top_countries_), {"Other", "India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "Turkey", "France"})
        out = fb.transform(pd.DataFrame([dict(Xtr.iloc[0], Country="Atlantis")]))
        self.assertEqual(out.Grouped_country.iloc[0], "Other")


class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = ModelService(ROOT / "models" / "model_bundle.joblib", ROOT / "baseline" / "Mental_Health_Model.pkl")

    def test_predict_valid_and_deterministic(self):
        a, b = self.svc.predict(GOOD), self.svc.predict(GOOD)
        self.assertEqual(a["predicted_mental_health_score"], b["predicted_mental_health_score"])
        s = a["predicted_mental_health_score"]
        self.assertTrue(3.0 <= s <= 10.0)
        iv = a["interval"]
        self.assertLessEqual(iv["lower"], s); self.assertGreaterEqual(iv["upper"], s)
        self.assertTrue(0 <= iv["lower"] and iv["upper"] <= 10 and iv["reliable"])
        self.assertIn(iv["uncertainty"], {"lower", "moderate", "higher"})
        self.assertIn("not a diagnosis", a["disclaimer"].lower())

    def test_interval_widens_with_level(self):
        w = [self.svc.predict(GOOD, level=l)["interval"] for l in (0.8, 0.9, 0.95)]
        self.assertLess(w[0]["upper"] - w[0]["lower"], w[1]["upper"] - w[1]["lower"])
        self.assertLess(w[1]["upper"] - w[1]["lower"], w[2]["upper"] - w[2]["lower"])
        with self.assertRaises(InputError): self.svc.predict(GOOD, level=0.5)

    def test_country_is_case_insensitive_and_unknown_is_grouped(self):
        self.assertEqual(self.svc.predict({**GOOD, "country": " india "})["predicted_mental_health_score"], self.svc.predict(GOOD)["predicted_mental_health_score"])
        r = self.svc.predict({**GOOD, "country": "Narnia"})
        self.assertTrue(any(w["field"] == "country" for w in r["input_warnings"]))
        self.assertEqual(r["predicted_mental_health_score"], self.svc.predict({**GOOD, "country": "Other"})["predicted_mental_health_score"])

    def test_out_of_domain_flagged_not_reliable(self):
        r = self.svc.predict({**GOOD, "age": 60})
        self.assertFalse(r["interval"]["reliable"]); self.assertTrue(any(w["field"] == "age" for w in r["input_warnings"]))

    def test_invalid_inputs_raise_field_errors(self):
        with self.assertRaises(InputError) as e: self.svc.predict({**GOOD, "gender": "Robot"})
        self.assertEqual(e.exception.errors[0]["field"], "gender")
        bad = dict(GOOD); del bad["age"]
        with self.assertRaises(InputError) as e: self.svc.predict(bad)
        self.assertEqual(e.exception.errors[0]["field"], "age")
        with self.assertRaises(InputError): self.svc.predict({**GOOD, "study_hours": "abc"})
        with self.assertRaises(InputError): self.svc.predict({**GOOD, "study_hours": float("nan")})

    def test_explanation_efficiency_and_fields(self):
        r = self.svc.predict(GOOD, explain=True)
        ex = r["explanation"]; self.assertEqual(len(ex["contributions"]), 12)
        self.assertAlmostEqual(sum(c["contribution"] for c in ex["contributions"]), ex["explained_prediction"] - ex["baseline_prediction"], delta=0.02)
        self.assertNotIn("cause", ex["method"].lower().replace("not causal", ""))

    def test_feedback_is_safe(self):
        low = self.svc.predict({**GOOD, "sleep_hours_per_night": 5.0, "stress_level": "Very High"})["feedback"]
        txt = " ".join(low).lower()
        self.assertIn("professional", txt)
        for banned in ("you are depressed", "you have depression", "diagnos" + "is:", "medication", "prescri"):
            self.assertNotIn(banned, txt)

    def test_sensitivity_labelled_non_causal(self):
        s = self.svc.sensitivity(GOOD); self.assertEqual(len(s["scenarios"]), 6); self.assertIn("not a causal", s["note"].lower())

    def test_baseline_model_preserved_and_matches_original_predictions(self):
        r = self.svc.predict_baseline(GOOD)
        import joblib
        b = joblib.load(ROOT / "baseline" / "Mental_Health_Model.pkl")
        row = pd.DataFrame([dict(Study_Hours=3.0, Age=21, Avg_Daily_Usage_Hours=5.0, Daily_Unlocks=170, Physical_Activity_Hours=1.7, Sleep_Hours_Per_Night=6.6,
                                 Stress_Level="Medium", Gender="Female", Academic_Level="Undergraduate", Most_Used_Platform="Instagram", Purpose_Of_Use="Entertainment", Grouped_country="India")])
        self.assertEqual(r["predicted_mental_health_score"], round(float(b.predict(row)[0]), 2))

    def test_model_info_and_metrics_are_measured_not_invented(self):
        info = self.svc.info(); rep = json.load(open(ROOT / "results" / "final_vs_baseline.json"))
        self.assertAlmostEqual(info["test_metrics"]["R2"], rep["improved"]["R2"], places=12)
        self.assertEqual(info["n_test"], 1500)

    def test_no_secrets_committed(self):
        for p in ROOT.rglob("*"):
            if p.suffix in {".py", ".js", ".html", ".md", ".yml", ".json"} and ".git" not in p.parts and "baseline" not in p.parts and p.is_file() and p.name != Path(__file__).name and p.stat().st_size < 2e6:
                t = p.read_text(errors="ignore")
                self.assertNotIn("sk-ant-", t, p); self.assertNotIn("BEGIN PRIVATE KEY", t, p)


if __name__ == "__main__":
    unittest.main()
