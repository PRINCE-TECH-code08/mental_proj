from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "Student_Social_Media_And_Mental_Health_Impact.csv"
BASELINE_PKL = ROOT / "baseline" / "Mental_Health_Model.pkl"
RESULTS, FIGURES, REPORTS, MODELS = (ROOT / d for d in ("results", "figures", "reports", "models"))

SEED = 42            # identical to the baseline notebook so the test split is identical
TEST_SIZE = 0.30     # identical to the baseline notebook
TARGET = "Mental_Health_Score"

RAW_INPUTS = ["Age", "Gender", "Country", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use",
              "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Study_Hours", "Physical_Activity_Hours",
              "Sleep_Hours_Per_Night", "Stress_Level"]
STRESS_ORDER = ["Low", "Medium", "High", "Very High"]

SKEWED = ["Study_Hours"]
NUMERIC = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
ORDINAL = ["Stress_Level"]
NOMINAL = ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use", "Grouped_country"]
ENGINEERED = ["Unlocks_Per_Screen_Hour", "Sleep_Deficit_Hours", "Study_Screen_Ratio", "Activity_Screen_Ratio"]

GROUPS = {
    "demographics": ["Age", "Gender", "Grouped_country", "Academic_Level"],
    "social_media": ["Most_Used_Platform", "Purpose_Of_Use", "Avg_Daily_Usage_Hours", "Daily_Unlocks"],
    "lifestyle": ["Study_Hours", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"],
    "stress": ["Stress_Level"],
}
BASE_FEATURES = [c for g in GROUPS.values() for c in g]
SUBGROUP_COLS = ["Gender", "Academic_Level", "Grouped_country", "Most_Used_Platform", "Stress_Level"]
