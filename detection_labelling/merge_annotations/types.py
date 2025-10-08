from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class AnnMergingContext:
    data_dir: Path
    models_dir: Path

    dataset_folder: str
    input_folders: str

    class_label: Dict[int, str]

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def output_dir(self) -> Path:
        return self.dataset_dir / "annotations_oob"

    def __post_init__(self):
        """Validate that required paths exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        for input_folder in self.input_folders:
            input_dir = self.dataset_dir / input_folder
            if not input_dir.exists():
                raise FileNotFoundError(f"Input folder not found: {input_dir}")

        """Validate that required paths exist."""
        self.output_dir.mkdir(exist_ok=True)
