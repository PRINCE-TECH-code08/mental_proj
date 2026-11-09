"""Feature construction and pipeline factory. Everything that learns from data lives INSIDE the Pipeline
so cross-validation cannot leak preprocessing statistics (incl. the country-bucketing list)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from . import config as C


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw user inputs -> model-ready frame. Learns the top-country list from TRAINING data only.
    n_top_countries=10 reproduces the baseline (its top-10 list contains the literal value 'Other')."""

    def __init__(self, n_top_countries: int = 10, engineered: bool = False):
        self.n_top_countries = n_top_countries
        self.engineered = engineered

    def fit(self, X, y=None):
        self.top_countries_ = X["Country"].value_counts().index[: self.n_top_countries].tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X["Grouped_country"] = X["Country"].where(X["Country"].isin(self.top_countries_), "Other")
        if self.engineered:
            eps = 0.1
            X["Unlocks_Per_Screen_Hour"] = X["Daily_Unlocks"] / (X["Avg_Daily_Usage_Hours"] + eps)
            X["Sleep_Deficit_Hours"] = (7.0 - X["Sleep_Hours_Per_Night"]).clip(lower=0)
            X["Study_Screen_Ratio"] = X["Study_Hours"] / (X["Avg_Daily_Usage_Hours"] + eps)
            X["Activity_Screen_Ratio"] = X["Physical_Activity_Hours"] / (X["Avg_Daily_Usage_Hours"] + eps)
        return X


def build_preprocessor(features):
    f = set(features)
    tr = []
    sk = [c for c in C.SKEWED if c in f]
    nu = [c for c in C.NUMERIC + C.ENGINEERED if c in f]
    od = [c for c in C.ORDINAL if c in f]
    no = [c for c in C.NOMINAL if c in f]
    if sk:
        tr.append(("skewed", Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), sk))
    if nu:
        tr.append(("numeric", StandardScaler(), nu))
    if od:
        tr.append(("ordinal", OrdinalEncoder(categories=[C.STRESS_ORDER]), od))
    if no:
        tr.append(("nominal", OneHotEncoder(handle_unknown="ignore"), no))
    assert len(sk + nu + od + no) == len(f), f"unmapped features: {f - set(sk + nu + od + no)}"
    return ColumnTransformer(tr)


def make_pipeline(model, features=None, engineered=False):
    features = list(features or C.BASE_FEATURES)
    return Pipeline([
        ("features", FeatureBuilder(engineered=engineered)),
        ("prep", build_preprocessor(features)),
        ("model", model),
    ])
