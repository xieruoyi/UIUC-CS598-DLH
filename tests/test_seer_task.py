from unittest.mock import MagicMock

import numpy as np
import pytest

from pyhealth.tasks.seer_survival_prediction import SEERSurvivalPrediction


@pytest.fixture
def task():
    """Returns a fresh instance of the task for each test."""
    return SEERSurvivalPrediction()


@pytest.fixture
def valid_patient():
    """Instantly conjures a fake PyHealth patient in memory."""
    mock_event = MagicMock()
    mock_event.attr_dict = {
        "label": 1,
        "age": 55,
        "year_dx": 2005,
        "race_White": 1,
        "stage_Localized": 1,
    }

    mock_patient = MagicMock()
    mock_patient.patient_id = "p1"
    mock_patient.get_events.return_value = [mock_event]
    return mock_patient


def test_seer_task_generates_samples(task, valid_patient) -> None:
    """Test that the SEER task generates valid samples."""
    samples = task(valid_patient)

    assert len(samples) == 1

    sample = samples[0]
    assert sample["patient_id"] == "p1"
    assert sample["visit_id"] == "p1_seer"
    assert "features" in sample
    assert "label" in sample


def test_seer_task_feature_extraction(task, valid_patient) -> None:
    """Test that task extracts features with the correct dimension."""
    samples = task(valid_patient)
    features = samples[0]["features"]

    # label is excluded, leaving exactly 4 feature columns
    assert features.shape[0] == 4
    assert isinstance(features, np.ndarray)


def test_seer_task_label_generation(task, valid_patient) -> None:
    """Test that labels are preserved as binary outputs."""
    samples = task(valid_patient)
    assert samples[0]["label"] == 1


def test_seer_task_feature_names_saved(task, valid_patient) -> None:
    """Test that feature names are saved consistently."""
    task(valid_patient)

    assert task.feature_names is not None
    assert "age" in task.feature_names
    assert "year_dx" in task.feature_names
    assert "label" not in task.feature_names


def test_seer_task_invalid_label_raises(task, valid_patient) -> None:
    """Test that a non-binary label raises a ValueError."""
    # Corrupt the label in memory
    valid_patient.get_events.return_value[0].attr_dict["label"] = 2

    with pytest.raises(ValueError, match="Label must be binary 0/1"):
        task(valid_patient)


def test_seer_task_non_numeric_feature_raises(task, valid_patient) -> None:
    """Test that a non-numeric feature raises a ValueError."""
    # Corrupt a feature in memory
    valid_patient.get_events.return_value[0].attr_dict["age"] = "bad_data"

    with pytest.raises(ValueError, match="Feature column"):
        task(valid_patient)