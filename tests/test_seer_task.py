from pathlib import Path

import pandas as pd
import pytest

from pyhealth.datasets import SEERDataset
from pyhealth.tasks.seer_survival_prediction import SEERSurvivalPrediction


def create_synthetic_seer_data(
    root: Path,
    rows: list[dict] | None = None,
) -> None:
    """Create a tiny synthetic SEER dataset for task testing."""
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if rows is None:
        rows = [
            {
                "patient_id": "p1",
                "event_time": "2005-01-01",
                "age": 55,
                "year_dx": 2005,
                "race_White": 1,
                "race_Black": 0,
                "stage_Localized": 1,
                "stage_Regional": 0,
                "label": 1,
            },
            {
                "patient_id": "p2",
                "event_time": "2006-01-01",
                "age": 63,
                "year_dx": 2006,
                "race_White": 0,
                "race_Black": 1,
                "stage_Localized": 0,
                "stage_Regional": 1,
                "label": 0,
            },
        ]

    df = pd.DataFrame(rows)
    csv_path = processed_dir / "seer_pyhealth.csv"
    df.to_csv(csv_path, index=False)

    yaml_text = """
version: "1.0"
tables:
  seer:
    file_path: processed/seer_pyhealth.csv
    patient_id: patient_id
    timestamp: event_time
    timestamp_format: "%Y-%m-%d"
    attributes:
      - age
      - year_dx
      - label
      - race_White
      - race_Black
      - stage_Localized
      - stage_Regional
""".strip()

    yaml_path = processed_dir / "seer.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")


def test_seer_task_generates_samples(tmp_path: Path) -> None:
    """Test that the SEER task generates valid samples."""
    create_synthetic_seer_data(tmp_path)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )
    task = SEERSurvivalPrediction()

    samples = dataset.set_task(task)

    assert len(samples) == 2

    sample = samples[0]
    assert "patient_id" in sample
    assert "visit_id" in sample
    assert "features" in sample
    assert "label" in sample


def test_seer_task_feature_extraction(tmp_path: Path) -> None:
    """Test that task extracts features with the correct dimension."""
    create_synthetic_seer_data(tmp_path)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )
    task = SEERSurvivalPrediction()

    samples = dataset.set_task(task)

    sample = samples[0]
    features = sample["features"]

    # label excluded, so remaining feature columns = 6
    assert features.shape[0] == 6


def test_seer_task_label_generation(tmp_path: Path) -> None:
    """Test that labels are preserved as binary outputs."""
    create_synthetic_seer_data(tmp_path)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )
    task = SEERSurvivalPrediction()

    samples = dataset.set_task(task)

    labels = {int(samples[i]["label"].item()) for i in range(len(samples))}
    assert labels == {0, 1}


def test_seer_task_feature_names_saved(tmp_path: Path) -> None:
    """Test that feature names are saved consistently."""
    create_synthetic_seer_data(tmp_path)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )
    task = SEERSurvivalPrediction()

    dataset.set_task(task)

    assert task.feature_names is not None
    assert "age" in task.feature_names
    assert "year_dx" in task.feature_names
    assert "label" not in task.feature_names


def test_seer_task_invalid_label_raises(tmp_path: Path) -> None:
    """Test that a non-binary label raises a ValueError."""
    rows = [
        {
            "patient_id": "p1",
            "event_time": "2005-01-01",
            "age": 55,
            "year_dx": 2005,
            "race_White": 1,
            "race_Black": 0,
            "stage_Localized": 1,
            "stage_Regional": 0,
            "label": 2,
        }
    ]
    create_synthetic_seer_data(tmp_path, rows=rows)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )
    task = SEERSurvivalPrediction()

    with pytest.raises(ValueError, match="Label must be binary 0/1"):
        dataset.set_task(task)


def test_seer_task_non_numeric_feature_raises(tmp_path: Path) -> None:
    """Test that a non-numeric feature raises a ValueError."""
    rows = [
        {
            "patient_id": "p1",
            "event_time": "2005-01-01",
            "age": "bad_value",
            "year_dx": 2005,
            "race_White": 1,
            "race_Black": 0,
            "stage_Localized": 1,
            "stage_Regional": 0,
            "label": 1,
        }
    ]
    create_synthetic_seer_data(tmp_path, rows=rows)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )
    task = SEERSurvivalPrediction()

    with pytest.raises(ValueError, match="Feature column"):
        dataset.set_task(task)