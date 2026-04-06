# DONOTMERGE Folder Documentation

This folder contains intermediate files, raw data, and results for the **SEER Breast Cancer 5-Year Survival Prediction** project. These files are not intended for merging into the main PyHealth repository, but rather serve as project artifacts for development and validation.

---

## File Inventory

### 1. **export.csv** (Raw Data Input, not in the repo since it is too large)
- **Type**: Raw SEER (Surveillance, Epidemiology, and End Results) dataset, filtered in SEER website (requires license)
- **Size**: Large (>50MB)
- **Content**: Unprocessed SEER breast cancer registry data with original column names and encoding
- **Purpose**: Source data for preprocessing pipeline
- **Columns**: Age recode, Sex, Race, Year of diagnosis, Histologic Type (ICD-O-3), Grade, Summary stage, Marital status, Laterality, LVI (Lymph-Vascular Invasion), ER/PR status, Survival months, SEER cause-specific death classification, and others
- **Note**: This is the raw export from SEER and requires cleaning, filtering, and transformation before use

---

### 2. **preprocess.py** (Data Preprocessing Script)
- **Type**: Python script (executable)
- **Purpose**: Main data preprocessing and feature engineering pipeline
- **Key Functions**:
  - `clean_string()`: Normalizes text fields and converts standard missing tokens to NaN
  - `parse_survival_months()`: Parses zero-padded survival month strings
  - `parse_age()`: Converts SEER age buckets (e.g., "75-79 years") to numeric midpoints
  - `build_event()`: Creates binary event indicator (cancer death vs. other/alive)
  - `simplify_grade()`: Maps verbose grade strings to standardized Grade I/II/III/IV/Unknown
  - `simplify_stage()`: Standardizes summary stage values
  - `simplify_laterality()`, `simplify_race()`, `simplify_marital()`, etc.: Categorical harmonization functions
  - `build_histology_category()`: Converts ICD-O-3 codes to categorical features; keeps top-20 frequencies separate, collapses rest to "Other"
  - `main()`: Orchestrates entire pipeline
- **Key Configurations**:
  - Diagnosis years: 2004-2009 (reproducibility cohort)
  - Prediction window: 60 months (5-year survival)
  - Female patients only: Yes
  - Histology top-n: 20
- **Outputs**: Generates `seer_clean_human_readable.csv` and `seer_ml_ready.csv` with comprehensive logs
- **Usage**: `python preprocess.py` (assumes `export.csv` exists in same directory)

---

### 3. **requirements.txt** (Python Dependencies)
- **Type**: pip-compatible dependency specification
- **Purpose**: Lists all Python packages needed for the SEER preprocessing and model training
- **Key Dependencies**:
  - Scientific stack: numpy, pandas, scikit-learn, scipy
  - Deep learning: torch (PyHealth backbone)
  - PyHealth ecosystem: accelerate, transformers, peft, einops, etc.
  - ML baselines: xgboost (for SEER survival models)
  - Data processing: pyarrow, polars, dask
  - Utilities: pyyaml, tqdm, requests
  - Visualization: matplotlib
  - Neuroscience (for PyHealth EEG support): mne
- **Note**: Includes all transitive dependencies needed to run preprocessing + PyHealth integration + ablation studies

---

### 4. **seer_clean_human_readable.csv** (Processed Data - Auditable Format)
- **Type**: CSV (tabular data)
- **Size**: ~288K rows (after filtering; original export has ~1.35M rows)
- **Purpose**: Cleaned SEER data in human-interpretable format with original feature definitions preserved
- **Columns**:
  - **Original/Audit columns**: `survival_months`, `event` (for auditing label construction; removed in ML-ready)
  - **Demographics**: `age` (numeric, midpoint of bucket), `year_dx` (diagnosis year)
  - **Clinical features**: `grade`, `stage`, `histology`, `laterality`, `er_status`, `pr_status`
  - **Sociodemographic**: `race`, `marital_status`
  - **Target**: `label` (binary: 1 = survived ≥60 months, 0 = cancer-specific death <60 months)
- **Filtering Applied**:
  - 2004-2009 diagnosis years only
  - Female patients only
  - Valid survival month values (>0, not null)
  - Unambiguous labels (survived ≥60 months OR death <60 months; excludes censored/alive with <60 months)
- **Missing Value Handling**: Numeric columns filled with median; categorical filled with "Unknown"
- **Purpose of Preservation**: Allows auditing: inspect `survival_months` and `event` to verify label construction logic

---

### 5. **seer_ml_ready.csv** (Processed Data - ML Format)
- **Type**: CSV (tabular data, one-hot encoded)
- **Size**: ~288K rows × 55 columns
- **Purpose**: Final ML-ready feature matrix for model training and ablation studies
- **Columns**:
  - **Numeric features**: `age`, `year_dx`
  - **One-hot encoded categorical features**: 
    - `race_*` (5 categories: American Indian/Alaska Native, Asian or Pacific Islander, Black, White, Unknown)
    - `histology_*` (21 categories: top 20 ICD-O-3 codes + 1 "Other" catch-all)
    - `grade_*` (5 categories: Grade I/II/III/IV, Unknown)
    - `stage_*` (5 categories: Localized, Regional, Distant, Unknown, Unknown/unstaged)
    - `marital_status_*` (6 categories)
    - `laterality_*` (4 categories: Left, Right, Bilateral, Unknown)
    - `er_status_*` (3 categories: Positive, Negative, Unknown)
    - `pr_status_*` (3 categories: Positive, Negative, Unknown)
  - **Target**: `label` (binary: 0 or 1)
- **Leakage Prevention**: Columns `survival_months` and `event` **removed** to prevent data leakage
- **Data Quality**: Rows = 288,820 (~21% of original 1.35M; due to filtering + label ambiguity removal)

---

### 6. **seer_preprocessing_summary.txt** (Processing Report)
- **Type**: Text documentation file
- **Purpose**: Metadata and audit trail of preprocessing decisions
- **Contents**:
  - List of generated output files
  - Configuration parameters (prediction window, diagnosis years, histology top-n)
  - Human-readable column definitions
  - ML-ready numeric and categorical column lists
  - Leakage column removal confirmation
- **Usage**: Quick reference to understand preprocessing configuration and feature schema

---

### 7. **seer.yaml** (PyHealth Dataset Schema)
- **Type**: YAML configuration file
- **Purpose**: PyHealth-compatible dataset schema for integration with the PyHealth framework
- **Key Sections**:
  - `version`: Schema version (1.0)
  - `tables`: Dataset table definition
    - `file_path`: Points to `seer_pyhealth.csv` (PyHealth-formatted version)
    - `patient_id`: Column name for unique patient identifier
    - `timestamp`: Column name for event timestamp (derived from `year_dx`)
    - `timestamp_format`: "%Y-%m-%d" (ISO 8601 format)
    - `attributes`: List of all feature and label columns to include
  - `join`: Empty (no inter-table joins needed for tabular data)
- **Usage**: Used by `SEERDataset` class to initialize PyHealth dataset loader
- **Integration**: Enables PyHealth's `set_task()` method to construct training samples

---

### 8. **seer_pyhealth.csv** (PyHealth-Formatted Data)
- **Type**: CSV (tabular data, one-hot encoded)
- **Purpose**: Intermediate dataset for PyHealth integration (transformed from `seer_ml_ready.csv`)
- **Additional Columns** (vs. `seer_ml_ready.csv`):
  - `patient_id`: Synthetic unique patient identifier (format: `seer_0`, `seer_1`, ...)
  - `visit_id`: Synthetic unique visit identifier (format: `visit_0`, `visit_1`, ...)
  - `event_time`: Timestamp derived from diagnosis year (format: "YYYY-01-01")
- **Purpose**: Provides patient/visit structure required by PyHealth's `BaseDataset`
- **Generated by**: `SEERDataset.prepare_metadata()` method during dataset initialization
- **Relationship to seer.yaml**: Referenced by `seer.yaml` as the data source

---

### 9. **feature_ablation_results.csv** (Ablation Study Results)
- **Type**: CSV (results table)
- **Purpose**: Quantitative comparison of model performance across feature sets for SEER survival prediction
- **Columns**:
  - `Feature Set`: Name of feature configuration (Full, Clinical, Minimal)
  - `Num Features`: Number of features used in training
  - `Accuracy`: Classification accuracy (proportion of correct predictions)
  - `F1`: F1-score (harmonic mean of precision and recall, useful for imbalanced data)
  - `AUROC`: Area Under the Receiver Operating Characteristic curve (probability of correct ranking)
- **Results Summary**:
  | Feature Set | Num Features | Accuracy | F1      | AUROC   |
  |-------------|-------------|----------|---------|---------|
  | Full        | 55          | 0.789    | 0.869   | 0.845   |
  | Clinical    | 34          | 0.786    | 0.867   | 0.843   |
  | Minimal     | 28          | 0.774    | 0.859   | 0.831   |
- **Interpretation**:
  - Full set (all features) provides best performance (AUROC = 0.845)
  - Clinical subset (age, diagnosis year, grade, stage, receptor status) achieves 99.5% of full performance while reducing features by 38%
  - Minimal subset (core demographics + stage/grade) still achieves 98.4% of full performance while reducing features by 49%
- **Implication**: Clinical judgment can guide feature selection; simpler models may be preferable for interpretability without significant performance loss

---

---

## Integration & Usage Files (PyHealth Framework Files)

These files integrate the SEER dataset into PyHealth and provide examples/tests. **Status**: ✅ **ACTIVELY USED** (not just for show).

### 10. **pyhealth/datasets/seer.py** (⭐ ACTIVELY USED)
- **Type**: Python class definition
- **Location**: `pyhealth/datasets/seer.py`
- **Purpose**: PyHealth dataset wrapper that loads and manages SEER data
- **Key Class**: `SEERDataset(BaseDataset)`
  - Inherits from PyHealth's `BaseDataset`
  - Auto-generates PyHealth-compatible data format on first load
  - Adds synthetic `patient_id`, `visit_id`, `event_time` columns if missing
  - Generates `seer_pyhealth.csv` and `seer.yaml` automatically
- **Key Methods**:
  - `__init__()`: Initializes dataset and calls `prepare_metadata()`
  - `prepare_metadata()`: Converts ML-ready CSV to PyHealth format with YAML schema
  - `info()`: Static method to display expected input format
- **Usage in Workflow**: Called by `examples/seer_survival_prediction_lr.py` and tests
- **Integration**: Exported in `pyhealth/datasets/__init__.py` for easy import: `from pyhealth.datasets import SEERDataset`
- **Status**: ✅ **Required for running examples and tests**

---

### 11. **pyhealth/tasks/seer_survival_prediction.py** (⭐ ACTIVELY USED)
- **Type**: Python class definition
- **Location**: `pyhealth/tasks/seer_survival_prediction.py`
- **Purpose**: Task wrapper for SEER 5-year survival binary classification
- **Key Class**: `SEERSurvivalPrediction(BaseTask)`
  - Inherits from PyHealth's `BaseTask`
  - Converts raw patient records → ML samples (features + binary labels)
  - Enforces binary label constraint (0 or 1)
  - Handles feature extraction and numeric validation
- **Key Attributes**:
  - `task_name: str = "SEERSurvivalPrediction"`
  - `input_schema = {"label": "binary"}`
  - `output_schema = {"label": "binary"}`
- **Key Methods**:
  - `__call__(patient)`: Transforms patient data into sample(s)
    - Extracts events from patient record
    - Validates label is numeric and binary
    - Extracts and orders features deterministically
    - Returns list of sample dictionaries with `features`, `label`, `patient_id`, `visit_id`
- **Usage in Workflow**: Created by `examples/seer_survival_prediction_lr.py`
- **Integration**: Imported in task workflows: `from pyhealth.tasks.seer_survival_prediction import SEERSurvivalPrediction`
- **Status**: ✅ **Required for running model training and dataset.set_task()**

---

### 12. **examples/seer_survival_prediction_lr.py** (⭐ ACTIVELY USED - Example Script)
- **Type**: Python executable script
- **Location**: `examples/seer_survival_prediction_lr.py`
- **Purpose**: Complete end-to-end example of SEER survival prediction with feature ablation
- **What It Does**:
  1. Loads `SEERDataset` from processed files
  2. Applies `SEERSurvivalPrediction` task to generate ML-ready samples
  3. Performs 80/20 train-test split
  4. **Baseline**: Trains Logistic Regression on all features
  5. **Ablation**: Removes one feature set (e.g., diagnosis year) and retrains to compare performance
  6. Reports Accuracy and AUROC scores
- **Command-Line Usage**:
  ```bash
  python examples/seer_survival_prediction_lr.py --root "path/to/PyHealth"
  ```
- **Outputs**: Prints model performance metrics (Accuracy, AUROC) for baseline and ablated models
- **Dependencies**: `SEERDataset`, `SEERSurvivalPrediction`, scikit-learn
- **Status**: ✅ **Example for users; demonstrates full workflow**
- **Key Features**:
  - Shows how to integrate custom datasets into PyHealth
  - Demonstrates train/test pipeline
  - Simple baseline for comparison with deep learning models

---

### 13. **tests/test_seer_dataset.py** (⭐ ACTIVELY USED - Unit Test)
- **Type**: Python pytest-compatible unit test
- **Location**: `tests/test_seer_dataset.py`
- **Purpose**: Validates `SEERDataset` class functionality
- **Test Cases**:
  - `test_seer_dataset_load()`: Verify dataset loads without errors
  - `test_seer_dataset_shape()`: Check output shape matches expectations
  - `test_seer_dataset_values()`: Validate data quality (no NaNs, correct ranges)
  - Synthetic data generation for lightweight testing
- **Usage**:
  ```bash
  pytest tests/test_seer_dataset.py -v
  ```
- **Dependencies**: pytest, pandas, `SEERDataset`
- **Status**: ✅ **Required for CI/CD and development validation**
- **Key Function**: `create_synthetic_seer_data(root)` generates tiny test dataset to avoid large file dependencies

---

### 14. **tests/test_seer_task.py** (⭐ ACTIVELY USED - Unit Test)
- **Type**: Python pytest-compatible unit test
- **Location**: `tests/test_seer_task.py`
- **Purpose**: Validates `SEERSurvivalPrediction` task functionality
- **Test Cases**:
  - `test_seer_task_call()`: Verify task correctly transforms patient data to samples
  - `test_seer_task_features()`: Check features are extracted in correct order
  - `test_seer_task_labels()`: Validate binary label constraint (0 or 1)
  - Error cases: Missing labels, non-numeric features, invalid label values
  - Synthetic data generation for lightweight testing
- **Usage**:
  ```bash
  pytest tests/test_seer_task.py -v
  ```
- **Dependencies**: pytest, `SEERDataset`, `SEERSurvivalPrediction`
- **Status**: ✅ **Required for CI/CD and development validation**
- **Key Function**: `create_synthetic_seer_data()` generates minimal test data with controllable rows

---

## File Usage Summary

### 📊 DONOTMERGE Folder (Data Artifacts)
| File | Used? | Purpose |
|------|-------|---------|
| `export.csv` | ✅ Input | Raw SEER data (must be obtained separately) |
| `preprocess.py` | ✅ Used | Converts raw→ML-ready; generates seer_ml_ready.csv |
| `requirements.txt` | ✅ Used | Python dependencies for entire project |
| `seer_ml_ready.csv` | ✅ Used | Input to dataset initialization |
| `seer_clean_human_readable.csv` | ✅ Show | Auditable format; not used algorithmically |
| `seer_preprocessing_summary.txt` | ✅ Show | Documentation only |
| `seer.yaml` | ✅ Used | PyHealth schema (auto-generated) |
| `seer_pyhealth.csv` | ✅ Used | PyHealth data format (auto-generated) |
| `feature_ablation_results.csv` | ✅ Show | Results/benchmarks |

### 📚 PyHealth Integration Files (Active Usage)
| File | Used? | Purpose |
|------|-------|---------|
| `pyhealth/datasets/seer.py` | ✅ **ACTIVE** | Core dataset wrapper; required for PyHealth integration |
| `pyhealth/tasks/seer_survival_prediction.py` | ✅ **ACTIVE** | Core task definition; required for model training |
| `examples/seer_survival_prediction_lr.py` | ✅ **ACTIVE** | End-to-end example; demonstrates full workflow |
| `tests/test_seer_dataset.py` | ✅ **ACTIVE** | Unit tests; required for CI/CD validation |
| `tests/test_seer_task.py` | ✅ **ACTIVE** | Unit tests; required for CI/CD validation |

### 🔄 Workflow Summary

1. **Raw Data** → `export.csv`
2. **Preprocessing** → `preprocess.py` executed → generates `seer_ml_ready.csv`
3. **Intermediate Outputs** → `seer_clean_human_readable.csv` (auditable), `seer_preprocessing_summary.txt` (report)
4. **ML-Ready Data** → `seer_ml_ready.csv` (no leakage, one-hot encoded)
5. **PyHealth Integration** → `seer_pyhealth.csv` + `seer.yaml` (auto-generated by `SEERDataset`)
6. **Dataset & Task** → `pyhealth/datasets/seer.py` + `pyhealth/tasks/seer_survival_prediction.py`
7. **Example Script** → `examples/seer_survival_prediction_lr.py` trains model
8. **Model Training & Ablation** → Train models with different feature sets
9. **Results** → `feature_ablation_results.csv` (performance comparison)

---

## Key Design Decisions

### Why These Configurations?
- **2004-2009 diagnosis years**: Ensures consistent follow-up time (end of study 2019 provides ≥10 years follow-up for all patients)
- **60-month (5-year) prediction window**: Standard in oncology; clinically meaningful milestone
- **Female-only cohort**: SEER breast cancer registry permits gender specificity; reduces confounding
- **Top-20 histology codes**: Balances feature count (avoids explosive one-hot expansion) with information preservation
- **One-hot encoding**: Treats categories as nominal (no implicit ordering); compatible with PyHealth's handling

### Why Remove `survival_months` and `event`?
- **Data leakage prevention**: Models must not see the outcome construction; e.g., if `survival_months >= 60` → `label=1`, seeing `survival_months` allows the model to trivially memorize this rule
- **Enforce proper learning**: Forces model to learn from clinical features (age, grade, stage, etc.) rather than outcome proxies

### Why Preserve Them in Human-Readable?
- **Audit trail**: Data scientists can verify label construction; e.g., check that all `label=0` cases have `event=1` AND `survival_months < 60`
- **Trust and reproducibility**: Critical for clinical applications where stakes are high

---

## How to Use These Files

### For Preprocessing:
```bash
python preprocess.py
```
Requires `export.csv` in same directory. Outputs `seer_ml_ready.csv` and `seer_clean_human_readable.csv`.

### For PyHealth Integration:
```python
from pyhealth.datasets import SEERDataset
from pyhealth.tasks import SEERSurvivalPrediction

dataset = SEERDataset(root="path/to/PyHealth")
task = SEERSurvivalPrediction()
dataset.set_task(task)
```

### For Ablation Studies:
Load `feature_ablation_results.csv` to compare model performance across feature sets.

---

## Notes

- This folder is marked **DONOTMERGE** because it contains large data files and results not suitable for production PyHealth releases
- For PyHealth contribution, only `preprocess.py`, `requirements.txt`, and integration code (`seer.py`, `seer_survival_prediction.py`) should be submitted
- Raw SEER data (`export.csv`) must be obtained separately from SEER website (Access requires registration)
