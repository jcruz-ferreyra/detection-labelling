from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class FramesDeduplicationContext:
    frames_dir: Path
    flann_index_params: Dict[str, Any]
    flann_search_params: Dict[str, Any]
    polygons_json_path: Path
    is_repeated_threshold: float = 0.05

    sift: Optional[cv2.SIFT] = None
    flann: Optional[cv2.FlannBasedMatcher] = None

    def __post_init__(self):
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Json file does not exist at: {self.frames_dir}")

        if not self.polygons_json_path.exists():
            raise FileNotFoundError(f"Json file does not exist at: {self.polygons_json_path}")
