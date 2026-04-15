from pathlib import Path

import pandas as pd
import pytest

from pyhealth.datasets import SEERDataset

# Define static paths for the config file
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "pyhealth" / "datasets" / "configs" / "seer.yaml"


def create_synthetic_seer_data(root: Path) -> None:
    """Create a synthetic ML-ready SEER dataset for testing."""
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "age": 55,
                "year_dx": 2005,
                "race_White": 1,
                "race_Black": 0,
                "stage_Localized": 1,
                "stage_Regional": 0,
                "label": 1,
            },
            {
                "age": 63,
                "year_dx": 2006,
                "race_White": 0,
                "race_Black": 1,
                "stage_Localized": 0,
                "stage_Regional": 1,
                "label": 0,
            },
            {
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


@pytest.fixture(scope="module")
def shared_seer_setup(tmp_path_factory):
    """Creates the data and initializes the dataset once for the whole module."""
    # Create a module-level temporary directory
    temp_dir = tmp_path_factory.mktemp("seer_test_dir")
    
    # Generate the synthetic data
    create_synthetic_seer_data(temp_dir)
    
    # Initialize PyHealth dataset
    dataset = SEERDataset(
        root=str(temp_dir),
        tables=["seer"],
        config_path=str(DEFAULT_CONFIG_PATH),
        dev=True,
        cache_dir=str(temp_dir)
    )
    
    return dataset, temp_dir


def test_seer_dataset_loads(shared_seer_setup) -> None:
    """Test that the synthetic SEER dataset initializes successfully."""
    dataset, temp_dir = shared_seer_setup

    assert dataset is not None
    assert Path(dataset.root) == temp_dir
    assert "seer" in dataset.tables


def test_seer_dataset_generates_pyhealth_csv(shared_seer_setup) -> None:
    """Test that dataset preparation generates the PyHealth-compatible CSV."""
    _, temp_dir = shared_seer_setup

    processed_dir = temp_dir / "processed"
    assert (processed_dir / "seer_pyhealth.csv").exists()


def test_seer_dataset_generated_csv_integrity(shared_seer_setup) -> None:
    """Test that the generated CSV preserves expected data columns and values."""
    _, temp_dir = shared_seer_setup

    generated_csv = temp_dir / "processed" / "seer_pyhealth.csv"
    df = pd.read_csv(generated_csv)

    assert "patient_id" in df.columns
    assert "visit_id" in df.columns
    assert "event_time" in df.columns
    assert list(df["patient_id"]) == ["seer_0", "seer_1", "seer_2"]

    assert len(df) == 3
    assert "age" in df.columns
    assert "year_dx" in df.columns
    assert "label" in df.columns

    row_p2 = df[df["patient_id"] == "seer_1"].iloc[0]
    assert row_p2["race_Black"] == 1
    assert row_p2["stage_Regional"] == 1
    assert row_p2["label"] == 0


def test_static_yaml_integrity() -> None:
    """Test that the static YAML in the repository contains expected fields."""
    yaml_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")

    assert 'version: "1.0"' in yaml_text
    assert "tables:" in yaml_text
    assert "seer:" in yaml_text
    assert "file_path: processed/seer_pyhealth.csv" in yaml_text
    assert "patient_id: patient_id" in yaml_text
    assert 'timestamp_format: "%Y-%m-%d"' in yaml_text
    assert "      - age" in yaml_text
    assert "      - year_dx" in yaml_text
    assert "      - label" in yaml_text