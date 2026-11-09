from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from . import config as C

_REG = {"Random Forest": RandomForestRegressor, "HistGradientBoosting": HistGradientBoostingRegressor,
        "Gradient Boosting": GradientBoostingRegressor}


def make_estimator(name: str, params: dict | None = None):
    return _REG[name](random_state=C.SEED, **(params or {}))
