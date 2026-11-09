"""Re-run the entire research pipeline end to end (about 15 minutes on 1 CPU). Usage: python scripts/run_all.py"""
import subprocess, sys, time
from pathlib import Path
S = ["01_data_audit", "02_baseline_reproduction", "03_model_selection", "04_ablation_and_features", "05_train_final",
     "06_error_analysis", "07_explainability", "08_uncertainty_eval", "09_fairness_subgroups", "10_sensitivity"]
for s in S:
    t = time.time(); print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / f"{s}.py")], check=True)
    print(f"   done in {time.time() - t:.0f}s", flush=True)
