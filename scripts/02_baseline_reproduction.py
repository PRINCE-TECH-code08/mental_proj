"""Phase 0/2: reproduce the ORIGINAL notebook exactly and score the ORIGINAL pickle on the identical split.
This is the fixed reference point for the paper's 'baseline vs improved' comparison."""
from _common import *
import joblib, numpy as np, pandas as pd
from mhp.metrics import reg_metrics
import sklearn

df = D.clean(D.load_raw())
# notebook computes top_countries on the FULL cleaned frame (before the split) -> replicate it exactly
top = df["Country"].value_counts().index[:10].tolist()
df["Grouped_country"] = df["Country"].where(df["Country"].isin(top), "Other")
cols = ["Study_Hours", "Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Physical_Activity_Hours",
        "Sleep_Hours_Per_Night", "Stress_Level", "Gender", "Academic_Level", "Most_Used_Platform",
        "Purpose_Of_Use", "Grouped_country"]
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(df[cols], df[C.TARGET], test_size=.3, random_state=42)
m = joblib.load(C.BASELINE_PKL)
out = dict(sklearn_used_here=sklearn.__version__, pickle_saved_with="1.9.1 (from InconsistentVersionWarning)",
           requirements_txt_pins="scikit-learn==1.9.0 (MISMATCH with pickle)",
           baseline_model=str(m.named_steps["random forest"]),
           notebook_reported=dict(R2_test=0.8780167461546369, MAE_test=0.34650236666666673, RMSE_test=0.462870, R2_train=0.9809069341581819),
           reproduced_test=reg_metrics(yte, m.predict(Xte)), reproduced_train=reg_metrics(ytr, m.predict(Xtr)),
           tuned_rf_notebook_reported=dict(R2_test=0.8652257547733472, MAE_test=0.36866505007865863),
           top_countries=top, n_train=len(Xtr), n_test=len(Xte))
out["reproduction_exact"] = bool(abs(out["reproduced_test"]["R2"] - out["notebook_reported"]["R2_test"]) < 1e-9)
save_json(out, C.RESULTS / "baseline_reproduction.json")
print(json.dumps({k: out[k] for k in ["reproduced_test", "reproduced_train", "reproduction_exact"]}, indent=1))
