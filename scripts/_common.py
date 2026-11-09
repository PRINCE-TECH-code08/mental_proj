import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "savefig.bbox": "tight"})
from mhp import config as C, data as D

for d in (C.RESULTS, C.FIGURES, C.REPORTS, C.MODELS):
    d.mkdir(exist_ok=True)


def load_split():
    df = D.clean(D.load_raw())
    return D.split(df)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
