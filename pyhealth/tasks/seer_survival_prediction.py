from typing import Dict, List
import numpy as np

from pyhealth.tasks import BaseTask


class SEERSurvivalPrediction(BaseTask):
    """SEER survival prediction task.

    Binary classification:
        label = 1 -> survived >= threshold
        label = 0 -> cancer death within threshold

    Assumes the source CSV has already produced a clean binary label column.
    """

    task_name: str = "SEERSurvivalPrediction"
    input_schema = {"label": "binary"}
    output_schema = {"label": "binary"}

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: List[str] | None = None

    def __call__(self, patient) -> List[Dict]:
        samples = []

        # Access events through patient.get_events(...)
        events = patient.get_events(event_type="seer")
        if len(events) == 0:
            return samples

        # One event per patient in this custom SEER table
        event = events[0]
        data = event.attr_dict

        if "label" not in data:
            raise KeyError(f"Missing 'label' for patient {patient.patient_id}")

        raw_label = data["label"]
        try:
            label = int(float(raw_label))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Label must be numeric 0/1, got {raw_label!r} "
                f"for patient {patient.patient_id}"
            ) from e

        if label not in (0, 1):
            raise ValueError(
                f"Label must be binary 0/1, got {label} "
                f"for patient {patient.patient_id}"
            )

        # Deterministic feature order
        feature_cols = sorted(k for k in data.keys() if k != "label")

        # Save once so example scripts can do named ablations
        if self.feature_names is None:
            self.feature_names = feature_cols

        features = []
        for k in feature_cols:
            v = data[k]
            try:
                features.append(float(v))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Feature column {k!r} must be numeric, got value {v!r} "
                    f"for patient {patient.patient_id}"
                ) from e

        features = np.asarray(features, dtype=np.float32)

        samples.append(
            {
                "patient_id": patient.patient_id,
                "visit_id": f"{patient.patient_id}_seer",
                "features": features,
                "label": label,
            }
        )

        return samples