from pathlib import Path

import pandas as pd

from pyhealth.datasets import SEERDataset


def create_synthetic_seer_data(root: Path) -> None:
    """Create a tiny synthetic SEER dataset for dataset testing."""
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
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
            {
                "patient_id": "p3",
                "event_time": "2007-01-01",
                "age": 47,
                "year_dx": 2007,
                "race_White": 1,
                "race_Black": 0,
                "stage_Localized": 1,
                "stage_Regional": 0,
                "label": 1,
            },
        ]
    )

    csv_path = processed_dir / "seer_pyhealth.csv"
    df.to_csv(csv_path, index=False)


def test_seer_dataset_loads(tmp_path: Path) -> None:
    """Test that the synthetic SEER dataset initializes successfully."""
    create_synthetic_seer_data(tmp_path)

    dataset = SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )

    assert dataset is not None
    assert Path(dataset.root) == tmp_path


def test_seer_dataset_generates_metadata_files(tmp_path: Path) -> None:
    """Test that dataset preparation generates CSV and YAML metadata files."""
    create_synthetic_seer_data(tmp_path)

    SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )

    processed_dir = tmp_path / "processed"
    assert (processed_dir / "seer_pyhealth.csv").exists()
    assert (processed_dir / "seer.yaml").exists()


def test_seer_dataset_generated_csv_integrity(tmp_path: Path) -> None:
    """Test that the generated CSV preserves expected data columns and values."""
    create_synthetic_seer_data(tmp_path)

    SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )

    generated_csv = tmp_path / "processed" / "seer_pyhealth.csv"
    df = pd.read_csv(generated_csv)

    assert len(df) == 3
    assert set(df["patient_id"]) == {"p1", "p2", "p3"}
    assert "age" in df.columns
    assert "year_dx" in df.columns
    assert "label" in df.columns

    row_p2 = df[df["patient_id"] == "p2"].iloc[0]
    assert row_p2["race_Black"] == 1
    assert row_p2["stage_Regional"] == 1
    assert row_p2["label"] == 0


def test_seer_dataset_generated_yaml_integrity(tmp_path: Path) -> None:
    """Test that the generated YAML contains expected schema fields."""
    create_synthetic_seer_data(tmp_path)

    SEERDataset(
        root=str(tmp_path),
        tables=["seer"],
    )

    yaml_path = tmp_path / "processed" / "seer.yaml"
    yaml_text = yaml_path.read_text(encoding="utf-8")

    assert 'version: "1.0"' in yaml_text
    assert "tables:" in yaml_text
    assert "seer:" in yaml_text
    assert "file_path: processed/seer_pyhealth.csv" in yaml_text
    assert "patient_id: patient_id" in yaml_text
    assert 'timestamp_format: "%Y-%m-%d"' in yaml_text
    assert "      - age" in yaml_text
    assert "      - year_dx" in yaml_text
    assert "      - label" in yaml_text