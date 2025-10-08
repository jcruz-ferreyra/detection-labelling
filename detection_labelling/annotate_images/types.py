from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


@dataclass
class ImageAnnotationContext:
    data_dir: Path
    models_dir: Path

    dataset_folder: Path
    yolo_name: Path
    yolo_params: Dict[str, Any]
    class_label: Dict[int, str]
    category_classes: Dict[str, list]
    class_confidence: List[Tuple[List[int], float]]

    model: Optional[YOLO] = None

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def yolo_path(self) -> Path:
        return self.models_dir / self.yolo_name

    def __post_init__(self):
        """Validate that required paths exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        if not self.yolo_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {self.yolo_path}")
