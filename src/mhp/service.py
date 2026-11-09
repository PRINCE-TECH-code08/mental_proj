"""Framework-free inference service (all API logic lives here so it can be unit-tested without FastAPI)."""
import bisect
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from . import config as C
from .explain import mc_shapley, display_value

FIELD_MAP = {c.lower(): c for c in C.RAW_INPUTS}      # api field name -> dataframe column
NUMERIC_FIELDS = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
HELP = ("This is an educational estimate, not a diagnosis. If you are struggling, please talk to someone you trust or a "
        "qualified health professional; in an emergency contact your local emergency services.")


class InputError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


class ModelService:
    def __init__(self, bundle_path=None, baseline_path=None):
        b = joblib.load(bundle_path or C.MODELS / "model_bundle.joblib")
        self.model, self.sigma, self.conf, self.ref, self.meta = b["model"], b["sigma_model"], b["conformal"], b["reference"], b["metadata"]
        self.background = pd.read_csv(Path(bundle_path).parent / "background.csv") if bundle_path else pd.read_csv(C.MODELS / "background.csv")
        self.baseline = joblib.load(baseline_path) if baseline_path else None
        nf = Path(bundle_path).parent / "noise_floor.json" if bundle_path else C.MODELS / "noise_floor.json"
        self.noise_floor = json.loads(nf.read_text()) if nf.exists() else None
        self._countries = {c.lower(): c for c in self.ref["top_countries"]}
        self._tq = self.ref["target"]["quantiles"]

    # ---------------- validation
    def to_frame(self, payload: dict):
        errs, row, warns = [], {}, []
        for api, col in FIELD_MAP.items():
            if api not in payload or payload[api] in (None, ""):
                errs.append(dict(field=api, message="required")); continue
            row[col] = payload[api]
        if errs:
            raise InputError(errs)
        row["Country"] = str(row["Country"]).strip()
        canon = self._countries.get(row["Country"].lower())
        if canon:
            row["Country"] = canon
        else:
            warns.append(dict(field="country", message=f"'{row['Country']}' is not one of the countries modelled individually; it is grouped as 'Other'."))
        for col, levels in self.ref["categorical"].items():
            if row[col] not in levels:
                errs.append(dict(field=col.lower(), message=f"must be one of {levels}"))
        for col in NUMERIC_FIELDS:
            try:
                row[col] = float(row[col])
            except (TypeError, ValueError):
                errs.append(dict(field=col.lower(), message="must be a number")); continue
            r = self.ref["numeric"][col]
            if not np.isfinite(row[col]):
                errs.append(dict(field=col.lower(), message="must be finite"))
            elif row[col] < r["min"] or row[col] > r["max"]:
                warns.append(dict(field=col.lower(), message=f"{row[col]:g} is outside the range in the training data ({r['min']:g}-{r['max']:g}); the prediction is an extrapolation and the interval is not reliable."))
        if errs:
            raise InputError(errs)
        return pd.DataFrame([row])[C.RAW_INPUTS], warns

    # ---------------- helpers
    def _percentile(self, score):
        return round(100 * bisect.bisect_left(self._tq, score) / (len(self._tq) - 1), 1)

    def _interval(self, df, pred, level):
        a = round(1 - level, 2)
        if str(a) not in self.conf["normalized_q"]:
            raise InputError([dict(field="level", message=f"must be one of {[round(1 - x, 2) for x in self.conf['alphas']]}")])
        sig = max(float(self.sigma.predict(df)[0]), self.conf["eps"])
        hw = self.conf["normalized_q"][str(a)] * sig
        hw90 = self.conf["normalized_q"]["0.1"] * sig
        t1, t2 = self.conf["uncertainty_thresholds_halfwidth90"]
        label = "lower" if hw90 <= t1 else ("moderate" if hw90 <= t2 else "higher")
        ghw = self.conf["global_q"][str(a)]
        return dict(level=level, lower=max(0.0, pred - hw), upper=min(10.0, pred + hw), half_width=hw,
                    global_lower=max(0.0, pred - ghw), global_upper=min(10.0, pred + ghw), uncertainty=label,
                    method="locally-weighted cross-conformal interval (see MODEL_CARD.md)")

    def feedback(self, row: pd.Series):
        r, out = self.ref["numeric"], []
        if row["Sleep_Hours_Per_Night"] < 7:
            out.append(f"Your reported sleep ({row['Sleep_Hours_Per_Night']:g} h) is below the 7-9 h range commonly recommended for adults. In the reference data, less sleep is associated with lower predicted scores.")
        if row["Avg_Daily_Usage_Hours"] > r["Avg_Daily_Usage_Hours"]["p75"]:
            out.append(f"Your reported screen time ({row['Avg_Daily_Usage_Hours']:g} h/day) is higher than about three quarters of the reference records; higher screen time is associated with lower predicted scores in this dataset.")
        if row["Physical_Activity_Hours"] < r["Physical_Activity_Hours"]["p25"]:
            out.append("Your reported physical activity is in the lowest quarter of the reference records; more activity is associated with higher predicted scores in this dataset.")
        if row["Stress_Level"] in ("High", "Very High"):
            out.append("You reported high stress. Higher reported stress is strongly associated with lower predicted scores in this dataset, which may partly reflect that the score itself is self-reported.")
        return out + [HELP]

    # ---------------- public API
    def predict(self, payload: dict, explain=False, level=0.9, n_perm=24):
        df, warns = self.to_frame(payload)
        pred = float(self.model.predict(df)[0])
        row = df.iloc[0]
        in_domain = not any(w["message"].find("outside the range") >= 0 for w in warns)
        iv = self._interval(df, pred, level)
        pct = self._percentile(pred)
        res = dict(predicted_mental_health_score=round(pred, 2), model_version=self.meta["model_version"],
                   interval=dict(level=iv["level"], lower=round(iv["lower"], 2), upper=round(iv["upper"], 2),
                                 global_lower=round(iv["global_lower"], 2), global_upper=round(iv["global_upper"], 2),
                                 uncertainty=iv["uncertainty"], reliable=in_domain, method=iv["method"]),
                   position=dict(percentile_in_reference=pct,
                                 description=("lower than most records in the reference dataset" if pct < 25 else "higher than most records in the reference dataset" if pct > 75 else "in the middle range of the reference dataset")),
                   input_warnings=warns, feedback=self.feedback(row),
                   disclaimer=self.meta["disclaimer"])
        if explain:
            phi, base, fx = mc_shapley(self.model.predict, row, self.background, C.RAW_INPUTS, n_perm=n_perm)
            top = sorted(phi.items(), key=lambda kv: -abs(kv[1]))
            res["explanation"] = dict(method="Monte-Carlo Shapley values (interventional, 60-row background). Describes model behaviour, not causal effects.",
                                      baseline_prediction=round(base, 3), explained_prediction=round(fx, 3),
                                      contributions=[dict(feature=k, value=display_value(row[k]), contribution=round(v, 3)) for k, v in top])
        return res

    def sensitivity(self, payload: dict):
        df, warns = self.to_frame(payload)
        base = float(self.model.predict(df)[0]); r = self.ref["numeric"]
        scen = [("Sleep +1 h", "Sleep_Hours_Per_Night", 1.0), ("Sleep -1 h", "Sleep_Hours_Per_Night", -1.0),
                ("Screen time +1 h", "Avg_Daily_Usage_Hours", 1.0), ("Screen time -1 h", "Avg_Daily_Usage_Hours", -1.0),
                ("Physical activity +0.5 h", "Physical_Activity_Hours", 0.5), ("Study +1 h", "Study_Hours", 1.0)]
        out = []
        for name, col, d in scen:
            d2 = df.copy(); d2[col] = float(np.clip(float(df[col].iloc[0]) + d, r[col]["min"], r[col]["max"]))
            out.append(dict(scenario=name, delta_prediction=round(float(self.model.predict(d2)[0]) - base, 3)))
        return dict(base_prediction=round(base, 2), scenarios=out, input_warnings=warns,
                    note="MODEL SENSITIVITY, not a causal effect: this shows how the model's output changes when one input is changed, holding the rest fixed. It does not show that changing your habits would change your wellbeing.")

    def predict_baseline(self, payload: dict):
        if self.baseline is None:
            raise RuntimeError("baseline model not loaded")
        df, warns = self.to_frame(payload)
        df["Grouped_country"] = df["Country"].where(df["Country"].isin(self.ref["top_countries"]), "Other")
        return dict(predicted_mental_health_score=round(float(self.baseline.predict(df[self.baseline.feature_names_in_])[0]), 2),
                    model="baseline (original notebook RandomForest)")

    def info(self):
        m = self.meta
        return {k: m[k] for k in ["model_version", "training_date", "algorithm", "features", "random_seed", "n_train", "n_test", "test_metrics", "interval_method", "disclaimer", "dataset_sha256"]} | \
               dict(training_domain=self.ref["numeric"], observed_target_range=[self.ref["target"]["min"], self.ref["target"]["max"]], noise_floor=self.noise_floor)
