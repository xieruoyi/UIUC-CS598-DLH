from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class SEERDataset(BaseDataset):
    """SEER breast cancer tabular dataset for survival prediction.

    This dataset is a thin PyHealth wrapper around a preprocessed tabular CSV.

    Expected primary input:
        root/processed/seer_ml_ready.csv

    For lightweight synthetic tests, it also accepts:
        root/processed/seer_pyhealth.csv

    During initialization, it will:
      1. read the input CSV
      2. add synthetic patient_id / visit_id if missing
      3. add a synthetic timestamp column derived from year_dx
      4. write a PyHealth-friendly CSV:
           root/processed/seer_pyhealth.csv
      5. write a matching YAML schema:
           root/processed/seer.yaml

    Then it calls BaseDataset with that generated schema.

    Notes:
      - Each row is treated as one patient and one visit.
      - The task file will decide which columns are inputs and which are labels.
      - This class does NOT rebuild labels; it only exposes the table to PyHealth.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        self.root = Path(root)

        generated_config = self.prepare_metadata(self.root)

        if config_path is None:
            config_path = generated_config

        if tables is None:
            tables = ["seer"]

        super().__init__(
            root=str(self.root),
            tables=tables,
            dataset_name=dataset_name or "seer",
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )
        

    @staticmethod
    def info() -> None:
        """Prints the expected input format for the SEER dataset."""
        print(
            "SEERDataset expects a preprocessed CSV at:\n"
            "  <root>/processed/seer_ml_ready.csv\n\n"
            "For synthetic tests, it also accepts:\n"
            "  <root>/processed/seer_pyhealth.csv\n\n"
            "Required columns:\n"
            "  - age\n"
            "  - year_dx\n"
            "  - label\n"
            "  - one-hot feature columns (race_*, grade_*, stage_*, etc.)\n\n"
            "The dataset wrapper will auto-create if missing:\n"
            "  - patient_id\n"
            "  - visit_id\n"
            "  - event_time\n"
            "  - processed/seer_pyhealth.csv\n"
            "  - processed/seer.yaml\n"
        )

    def prepare_metadata(self, root: Path) -> Path:
        """Prepare a PyHealth-compatible CSV and YAML schema.

        Args:
            root: Project root directory containing processed data.

        Returns:
            Path to the generated YAML config.
        """
        processed_dir = root / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        source_csv = processed_dir / "seer_ml_ready.csv"
        output_csv = processed_dir / "seer_pyhealth.csv"
        output_yaml = processed_dir / "seer.yaml"

        # Allow synthetic test data to bypass the full preprocessing pipeline.
        if not source_csv.exists():
            alt_csv = processed_dir / "seer_pyhealth.csv"
            if alt_csv.exists():
                source_csv = alt_csv

        if not source_csv.exists():
            raise FileNotFoundError(
                f"Could not find {processed_dir / 'seer_ml_ready.csv'} or "
                f"{processed_dir / 'seer_pyhealth.csv'}. "
                "Run preprocessing first, or provide synthetic test data."
            )

        df = pd.read_csv(source_csv)

        if "patient_id" not in df.columns:
            df.insert(0, "patient_id", [f"seer_{i}" for i in range(len(df))])

        if "visit_id" not in df.columns:
            df.insert(1, "visit_id", [f"visit_{i}" for i in range(len(df))])

        if "event_time" not in df.columns:
            if "year_dx" not in df.columns:
                raise KeyError(
                    "Missing required column 'year_dx' needed to synthesize "
                    "'event_time'."
                )
            year_series = (
                pd.to_numeric(df["year_dx"], errors="coerce")
                .fillna(2000)
                .astype(int)
            )
            df.insert(2, "event_time", year_series.astype(str) + "-01-01")

        df.to_csv(output_csv, index=False)

        excluded = {"patient_id", "visit_id", "event_time"}
        attributes = [c for c in df.columns if c not in excluded]

        yaml_text = self._build_yaml(
            file_path="processed/seer_pyhealth.csv",
            patient_id="patient_id",
            timestamp="event_time",
            timestamp_format="%Y-%m-%d",
            attributes=attributes,
        )
        output_yaml.write_text(yaml_text, encoding="utf-8")

        logger.info("Generated PyHealth SEER table at %s", output_csv)
        logger.info("Generated PyHealth SEER config at %s", output_yaml)

        return output_yaml

    @staticmethod
    def _build_yaml(
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