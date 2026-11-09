"""Phase 8/15/16: fit the final model on TRAIN, calibrate cross-conformal intervals from out-of-fold residuals,
evaluate ONCE on the untouched test set against the ORIGINAL baseline pickle, and write the deployable bundle."""
import datetime, hashlib, platform
from _common import *
import joblib, numpy as np, pandas as pd, sklearn, scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from mhp.features import make_pipeline
from mhp.models import make_estimator
from mhp.metrics import reg_metrics, bootstrap_ci
from mhp import __version__

Xtr, Xte, ytr, yte = load_split()
sel = json.load(open(C.RESULTS / "selected_model.json")); dec = json.load(open(C.RESULTS / "feature_decision.json"))
eng = dec["use_engineered_features"]; feats = C.BASE_FEATURES + (C.ENGINEERED if eng else [])
print("final:", sel["selected"], sel["params"], "engineered:", eng)

model = make_pipeline(make_estimator(sel["selected"], sel["params"]), feats, eng).fit(Xtr, ytr)
pred = model.predict(Xte)

# ---- original baseline on identical rows
base = joblib.load(C.BASELINE_PKL)
top = D.clean(D.load_raw())["Country"].value_counts().index[:10].tolist()   # exactly as the notebook did
Xb = Xte.copy(); Xb["Grouped_country"] = Xb["Country"].where(Xb["Country"].isin(top), "Other")
pred_b = base.predict(Xb[base.feature_names_in_])
# ---- baseline configuration re-fitted inside the new leak-free pipeline (isolates pickle/ordering noise)
rf0 = make_pipeline(make_estimator("Random Forest", {}), C.BASE_FEATURES, False).fit(Xtr, ytr); pred_r = rf0.predict(Xte)

m_new, m_base, m_rf0 = reg_metrics(yte, pred), reg_metrics(yte, pred_b), reg_metrics(yte, pred_r)
y = yte.values; n = len(y)
d_mae = lambda i: np.abs(y[i] - pred_b[i]).mean() - np.abs(y[i] - pred[i]).mean()          # >0 => improved better
r2 = lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
d_r2 = lambda i: r2(y[i], pred[i]) - r2(y[i], pred_b[i])
cmp_ = dict(baseline_original_pickle=m_base, baseline_config_refit=m_rf0, improved=m_new,
            delta_MAE_baseline_minus_improved=d_mae(np.arange(n)), delta_MAE_CI95=bootstrap_ci(d_mae, n),
            delta_R2_improved_minus_baseline=d_r2(np.arange(n)), delta_R2_CI95=bootstrap_ci(d_r2, n),
            note="Paired bootstrap over the 1,500 test rows (2000 resamples). CI excluding 0 => difference not explained by test-set sampling alone; it does not account for training-set variability.")
save_json(cmp_, C.RESULTS / "final_vs_baseline.json")
tp = Xte.copy(); tp["y_true"] = y; tp["pred_improved"] = pred; tp["pred_baseline"] = pred_b
tp.to_csv(C.RESULTS / "test_predictions.csv", index=True, index_label="row_id")

# ---- cross-conformal calibration on OUT-OF-FOLD training residuals
kf = KFold(5, shuffle=True, random_state=C.SEED)
oof = cross_val_predict(make_pipeline(make_estimator(sel["selected"], sel["params"]), feats, eng), Xtr, ytr, cv=kf)
res = np.abs(ytr.values - oof)
mk_sigma = lambda: make_pipeline(RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=C.SEED), feats, eng)
sig_oof = cross_val_predict(mk_sigma(), Xtr, res, cv=KFold(5, shuffle=True, random_state=C.SEED + 7))
sigma = mk_sigma().fit(Xtr, res)
EPS = 0.05
sig_oof = np.maximum(sig_oof, EPS)
def q(scores, a):
    k = int(np.ceil((len(scores) + 1) * (1 - a))); return float(np.sort(scores)[min(k, len(scores)) - 1])
alphas = [0.2, 0.1, 0.05]
conf = dict(method="cross-conformal on out-of-fold residuals (5-fold), global and locally-weighted",
            eps=EPS, alphas=alphas, n_calibration=len(res),
            global_q={str(a): q(res, a) for a in alphas}, normalized_q={str(a): q(res / sig_oof, a) for a in alphas})
hw = conf["normalized_q"]["0.1"] * sig_oof
conf["uncertainty_thresholds_halfwidth90"] = [float(np.percentile(hw, 33.3)), float(np.percentile(hw, 66.7))]
save_json(conf, C.MODELS / "conformal.json")

# ---- reference stats for OOD checks / feedback, and Shapley background
num = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
ref = dict(numeric={c: dict(min=float(Xtr[c].min()), max=float(Xtr[c].max()), p05=float(Xtr[c].quantile(.05)), p25=float(Xtr[c].quantile(.25)),
                            median=float(Xtr[c].median()), p75=float(Xtr[c].quantile(.75)), p95=float(Xtr[c].quantile(.95))) for c in num},
           categorical={c: sorted(Xtr[c].unique().tolist()) for c in ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use", "Stress_Level"]},
           target=dict(min=float(ytr.min()), max=float(ytr.max()), mean=float(ytr.mean()), quantiles=[float(v) for v in np.quantile(ytr, np.linspace(0, 1, 101))]),
           top_countries=model["features"].top_countries_)
save_json(ref, C.MODELS / "reference.json")
Xtr[C.RAW_INPUTS].sample(60, random_state=C.SEED).to_csv(C.MODELS / "background.csv", index=False)

meta = dict(model_version=__version__, training_date=datetime.date.today().isoformat(), algorithm=sel["selected"], hyperparameters=sel["params"],
            engineered_features=eng, features=feats, raw_inputs=C.RAW_INPUTS, random_seed=C.SEED, split=f"train_test_split(test_size={C.TEST_SIZE}, random_state={C.SEED})",
            dataset_sha256=hashlib.sha256(C.RAW_CSV.read_bytes()).hexdigest(), dataset_rows_clean=len(Xtr) + len(Xte), n_train=len(Xtr), n_test=len(Xte),
            test_metrics=m_new, baseline_test_metrics=m_base,
            libs=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
            interval_method=conf["method"],
            disclaimer="Predicts a dataset-specific wellbeing score from self-reported habits. Not a diagnosis. Associations are not causal.")
save_json(meta, C.MODELS / "metadata.json")
joblib.dump(dict(model=model, sigma_model=sigma, conformal=conf, reference=ref, metadata=meta), C.MODELS / "model_bundle.joblib", compress=3)
print(json.dumps(cmp_, indent=1)); print("bundle MB:", round((C.MODELS / "model_bundle.joblib").stat().st_size / 1e6, 1))
