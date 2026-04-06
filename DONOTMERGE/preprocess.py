import os
import re
from typing import Iterable, List

import numpy as np
import pandas as pd


INPUT_CSV = "export.csv"
OUTPUT_DIR = "processed"

AGE_COL = "Age recode with <1 year olds and 90+"
SEX_COL = "Sex"
RACE_COL = "Race recode (W, B, AI, API)"
YEAR_COL = "Year of diagnosis"
HIST_COL = "Histologic Type ICD-O-3"
GRADE_COL = "Grade Recode (thru 2017)"
STAGE_COL = "Summary stage 2000 (1998-2017)"
MARITAL_COL = "Marital status at diagnosis"
LATERALITY_COL = "Laterality"
LVI_COL = "Lymph-vascular Invasion (2004+ varying by schema)"
ER_COL = "ER Status Recode Breast Cancer (1990+)"
PR_COL = "PR Status Recode Breast Cancer (1990+)"
SURV_COL = "Survival months"
CAUSE_COL = "SEER cause-specific death classification"

# Cohort / task configuration
DIAGNOSIS_YEAR_MIN = 2004
DIAGNOSIS_YEAR_MAX = 2009
PREDICTION_WINDOW_MONTHS = 60
KEEP_ONLY_FEMALE = True

# Histology handling:
# Keep the most common histology codes as separate one-hot categories and collapse
# the rest into "Other". This avoids treating ICD-O-3 codes as numeric and also
# prevents a huge sparse design matrix.
HISTOLOGY_TOP_N = 20

MISSING_TOKENS = {
    "Blank(s)",
    "Unknown",
    "Recode not available",
    "Not available",
    "Not applicable",
    "",
}

NON_CANCER_OR_ALIVE = "Alive or dead of other cause"
CANCER_DEATH = "Dead (attributable to this cancer dx)"


def clean_string(x):
    """Normalize strings and turn standard missing-like tokens into NaN."""
    if pd.isna(x):
        return np.nan
    if not isinstance(x, str):
        return x
    x = x.strip()
    if x in MISSING_TOKENS:
        return np.nan
    return x


def parse_survival_months(x):
    """Parse zero-padded survival month strings like '0014' -> 14."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    return np.nan


def parse_age(x):
    """
    Convert SEER age buckets to a single numeric value.

    Examples:
        '75-79 years' -> 77.0
        '90+ years'   -> 90.0
        '<1 year'     -> 0.5
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        return (lo + hi) / 2.0

    if "90+" in s:
        return 90.0

    if "<1" in s:
        return 0.5

    return np.nan


def build_event(x):
    """
    event = 1 if death attributable to this cancer
    event = 0 if alive or dead of other cause
    event = NaN otherwise
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s == CANCER_DEATH:
        return 1
    if s == NON_CANCER_OR_ALIVE:
        return 0

    return np.nan


def simplify_grade(x):
    """Map verbose grade strings to Grade I/II/III/IV/Unknown."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).lower()

    if "grade iv" in s or "undifferentiated" in s or "anaplastic" in s:
        return "Grade IV"
    if "grade iii" in s or "poorly differentiated" in s:
        return "Grade III"
    if "grade ii" in s or "moderately differentiated" in s:
        return "Grade II"
    if "grade i" in s or "well differentiated" in s:
        return "Grade I"

    return "Unknown"


def simplify_stage(x):
    """Map summary stage values to a small clean set."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()
    if s in {"Localized", "Regional", "Distant", "Unknown/unstaged"}:
        return s
    return "Unknown"


def simplify_laterality(x):
    """Map laterality values to Left/Right/Bilateral/Unknown."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()
    if s.startswith("Left"):
        return "Left"
    if s.startswith("Right"):
        return "Right"
    if "Bilateral" in s:
        return "Bilateral"
    return "Unknown"


def simplify_binary_recode(x):
    """Map receptor-like variables to Positive/Negative/Unknown."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()
    if s == "Positive":
        return "Positive"
    if s == "Negative":
        return "Negative"
    return "Unknown"


def simplify_lvi(x):
    """Map LVI values to Present/Absent/Unknown."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip().lower()

    if "present" in s or s == "yes":
        return "Present"
    if "absent" in s or s == "no":
        return "Absent"
    return "Unknown"


def simplify_race(x):
    """Standardize race values to a compact, stable set."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()
    allowed = {
        "White",
        "Black",
        "American Indian/Alaska Native",
        "Asian or Pacific Islander",
    }
    if s in allowed:
        return s
    return "Unknown"


def simplify_marital(x):
    """Keep the main marital status buckets and collapse anything else to Unknown."""
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()
    allowed = {
        "Married (including common law)",
        "Single (never married)",
        "Divorced",
        "Widowed",
        "Separated",
        "Unmarried or Domestic Partner",
    }
    if s in allowed:
        return s
    return "Unknown"


def build_histology_category(series: pd.Series, top_n: int = HISTOLOGY_TOP_N) -> pd.Series:
    """
    Convert ICD-O-3 histology codes into categorical strings suitable for one-hot encoding.

    The most frequent `top_n` codes are kept as individual categories; the rest are grouped
    into 'Other'. Missing stays as 'Unknown'.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    as_str = numeric.astype("Int64").astype(str)
    as_str = as_str.replace("<NA>", "Unknown")

    value_counts = as_str[as_str != "Unknown"].value_counts()
    top_codes = set(value_counts.head(top_n).index.tolist())

    def collapse(code: str) -> str:
        if code == "Unknown":
            return "Unknown"
        if code in top_codes:
            return f"Histology_{code}"
        return "Histology_Other"

    return as_str.apply(collapse)


def ensure_required_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# Moved yaml script into preprocess, shouldn't be generated in pyhealth release
def build_yaml(
    file_path: str,
    patient_id: str,
    timestamp: str,
    timestamp_format: str,
    attributes: List[str],
) -> str:
    """Build a minimal custom-dataset YAML config."""
    attr_lines = "\n".join([f"      - {col}" for col in attributes])

    return (
        f'version: "1.0"\n'
        f"tables:\n"
        f"  seer:\n"
        f"    file_path: {file_path}\n"
        f"    patient_id: {patient_id}\n"
        f"    timestamp: {timestamp}\n"
        f'    timestamp_format: "{timestamp_format}"\n'
        f"    attributes:\n"
        f"{attr_lines}\n"
        f"    join: []\n"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print("Raw shape:", df.shape)

    required_cols = [
        AGE_COL,
        SEX_COL,
        RACE_COL,
        YEAR_COL,
        HIST_COL,
        GRADE_COL,
        STAGE_COL,
        MARITAL_COL,
        LATERALITY_COL,
        LVI_COL,
        ER_COL,
        PR_COL,
        SURV_COL,
        CAUSE_COL,
    ]
    ensure_required_columns(df, required_cols)

    # Normalize text fields; only map object columns for speed.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(clean_string)

    # Parse core variables
    df[SURV_COL] = df[SURV_COL].apply(parse_survival_months)
    df["year_dx"] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df["age"] = df[AGE_COL].apply(parse_age)
    df["event"] = df[CAUSE_COL].apply(build_event)

    # Basic validity filters
    df = df[df[SURV_COL].notna()]
    df = df[df[SURV_COL] > 0]
    df = df[df["year_dx"].notna()]

    if KEEP_ONLY_FEMALE:
        df = df[df[SEX_COL] == "Female"]

    # Reproducibility cohort
    df = df[
        (df["year_dx"] >= DIAGNOSIS_YEAR_MIN)
        & (df["year_dx"] <= DIAGNOSIS_YEAR_MAX)
    ]
    print(
        f"After year filtering ({DIAGNOSIS_YEAR_MIN}-{DIAGNOSIS_YEAR_MAX}): "
        f"{len(df)} rows"
    )

    # Simplify categories
    df["race_simple"] = df[RACE_COL].apply(simplify_race)
    df["grade_simple"] = df[GRADE_COL].apply(simplify_grade)
    df["stage_simple"] = df[STAGE_COL].apply(simplify_stage)
    df["marital_simple"] = df[MARITAL_COL].apply(simplify_marital)
    df["laterality_simple"] = df[LATERALITY_COL].apply(simplify_laterality)
    df["er_simple"] = df[ER_COL].apply(simplify_binary_recode)
    df["pr_simple"] = df[PR_COL].apply(simplify_binary_recode)
    df["lvi_simple"] = df[LVI_COL].apply(simplify_lvi)
    df["histology_category"] = build_histology_category(df[HIST_COL])

    # Vectorized 5-year label construction:
    # label = 1 if survived >= 60 months
    # label = 0 if cancer-specific death before 60 months
    # otherwise exclude as ambiguous
    df["label"] = np.nan
    df.loc[df[SURV_COL] >= PREDICTION_WINDOW_MONTHS, "label"] = 1
    df.loc[
        (df[SURV_COL] < PREDICTION_WINDOW_MONTHS) & (df["event"] == 1),
        "label"
    ] = 0

    before_label_filter = len(df)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    print(
        f"After removing ambiguous labels: {len(df)} rows "
        f"(dropped {before_label_filter - len(df)})"
    )

    # Human-readable cleaned table: keep label-construction columns here for auditing.
    clean_df = pd.DataFrame(
        {
            "age": df["age"],
            "year_dx": df["year_dx"].astype(int),
            "survival_months": df[SURV_COL].astype(int),
            "event": df["event"].astype("Int64"),
            "race": df["race_simple"],
            "histology": df["histology_category"],
            "grade": df["grade_simple"],
            "stage": df["stage_simple"],
            "marital_status": df["marital_simple"],
            "laterality": df["laterality_simple"],
            "er_status": df["er_simple"], 
            "pr_status": df["pr_simple"],
            "label": df["label"],
        }
    )

    if (df["lvi_simple"] != "Unknown").sum() > 0:
        clean_df["lvi_status"] = df["lvi_simple"]

    # Fill missing numeric values
    numeric_cols = ["age", "year_dx", "survival_months"]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # Keep event as nullable integer for auditing only
    clean_df["event"] = clean_df["event"].astype("Int64")

    # Fill missing categorical values
    clean_cat_cols = [
        c for c in clean_df.columns
        if c not in numeric_cols + ["event", "label"]
    ]
    for col in clean_cat_cols:
        clean_df[col] = clean_df[col].fillna("Unknown")

    human_path = os.path.join(OUTPUT_DIR, "seer_clean_human_readable.csv")
    clean_df.to_csv(human_path, index=False)

    # FINAL ML-READY TABLE:
    # remove leakage columns before one-hot encoding
    model_df = clean_df.drop(columns=["survival_months", "event"])

    model_numeric_cols = ["age", "year_dx"]
    model_cat_cols = [c for c in model_df.columns if c not in model_numeric_cols + ["label"]]

    ml_df = pd.get_dummies(
        model_df,
        columns=model_cat_cols,
        drop_first=False,
        dtype=int,
    )

    ml_path = os.path.join(OUTPUT_DIR, "seer_ml_ready.csv")
    ml_df.to_csv(ml_path, index=False)

    # FINAL PYHEALTH DATA
    # Originally prepare_metadata() in seer.py
    pyhealth_df = ml_df.copy()
    
    # 1. Inject patient_id and visit_id
    pyhealth_df.insert(0, "patient_id", [f"seer_{i}" for i in range(len(pyhealth_df))])
    pyhealth_df.insert(1, "visit_id", [f"visit_{i}" for i in range(len(pyhealth_df))])

    # 2. Synthesize event_time from year_dx
    year_series = (
        pd.to_numeric(pyhealth_df["year_dx"], errors="coerce")
        .fillna(2000)
        .astype(int)
    )
    pyhealth_df.insert(2, "event_time", year_series.astype(str) + "-01-01")

    # 3. Save PyHealth CSV
    pyhealth_path = os.path.join(OUTPUT_DIR, "seer_pyhealth.csv")
    pyhealth_df.to_csv(pyhealth_path, index=False)

    # 4. Generate and save PyHealth YAML config
    excluded = {"patient_id", "visit_id", "event_time"}
    attributes = [c for c in pyhealth_df.columns if c not in excluded]

    yaml_text = build_yaml(
        file_path=os.path.join(OUTPUT_DIR, "seer_pyhealth.csv"),
        patient_id="patient_id",
        timestamp="event_time",
        timestamp_format="%Y-%m-%d",
        attributes=attributes,
    )
    yaml_path = os.path.join(OUTPUT_DIR, "seer.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
        
    # Summary file
    schema_lines = [
        "Generated files:",
        f"- {human_path}",
        f"- {ml_path}",
        f"- {pyhealth_path}",
        f"- {yaml_path}",
        "",
        f"Prediction window (months): {PREDICTION_WINDOW_MONTHS}",
        f"Diagnosis years: {DIAGNOSIS_YEAR_MIN}-{DIAGNOSIS_YEAR_MAX}",
        f"Histology top-n kept separate: {HISTOLOGY_TOP_N}",
        "",
        "Human-readable columns:",
        *[f"- {c}" for c in clean_df.columns],
        "",
        "ML-ready numeric columns:",
        *[f"- {c}" for c in model_numeric_cols],
        "",
        "ML-ready categorical columns one-hot encoded:",
        *[f"- {c}" for c in model_cat_cols],
        "",
        "Leakage columns removed from ML-ready file:",
        "- survival_months",
        "- event",
    ]
    schema_path = os.path.join(OUTPUT_DIR, "seer_preprocessing_summary.txt")
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("\n".join(schema_lines))

    print("Cleaned shape:", clean_df.shape)
    print("ML-ready shape:", ml_df.shape)
    print("Saved:", human_path)
    print("Saved:", ml_path)
    print("Saved:", pyhealth_path)
    print("Saved:", yaml_path)
    print("Saved:", schema_path)

    print("\nLabel distribution:")
    print(clean_df["label"].value_counts(dropna=False))
    print("\nLabel ratio:")
    print(clean_df["label"].value_counts(normalize=True, dropna=False))

    print("\nML-ready first columns:")
    print(ml_df.columns[:20].tolist())

    print("\nLeakage check:")
    print("survival_months in ml_df:", "survival_months" in ml_df.columns)
    print("event in ml_df:", "event" in ml_df.columns)


if __name__ == "__main__":
    main()