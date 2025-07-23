from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


@dataclass
class ImageAnnotationContext:
    dataset_dir: Path
    yolo_path: Path
    yolo_params: Dict[str, Any]
    class_label: Dict[int, str]
    category_classes: Dict[str, list]
    class_confidence: List[Tuple[List[int], float]]

    model: Optional[YOLO] = None
