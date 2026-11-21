"""Persist and reload experiment results so plots come from saved files."""
import json
from pathlib import Path

import pandas as pd

from driftlens.experiment import ExperimentResult


class ResultStore:
    """CSV (one per table) + metadata.json inside a directory."""

    def __init__(self, directory) -> None:
        self.directory = Path(directory)

    def save(self, result: ExperimentResult) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # List comprehension used for its side effect: write one CSV per table.
        _ = [frame.to_csv(self.directory / f"{name}.csv", index=False) for name, frame in result.tables.items()]
        (self.directory / "metadata.json").write_text(json.dumps(result.metadata, indent=2, default=str))

    def exists(self) -> bool:
        return (self.directory / "metadata.json").exists()

    def load(self) -> ExperimentResult:
        tables = {path.stem: pd.read_csv(path) for path in sorted(self.directory.glob("*.csv"))}
        metadata = json.loads((self.directory / "metadata.json").read_text())
        return ExperimentResult(tables, metadata)
