from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import supervision as sv
from ultralytics import YOLO


@dataclass
class VideoProcessingContext:
    video_path: Path
    output_dir: Path
    model_name: str
    model_params: Dict[str, Any]
    class_label: Dict[int, str]
    category_classes: Dict[str, list]
    class_confidence: List[Tuple[List[int], float]]
    frame_handler_params: Dict[str, Any]
    quadrant_scorer_params: Dict[str, Any]

    model: Optional[YOLO] = None
    generator: Optional[Generator] = None
    video_info: Optional[sv.VideoInfo] = None
    frame_handler: Optional["FrameHandler"] = None
    quadrant_scorer: Optional["QuadrantScorer"] = None

    saved_frames: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist at: {self.video_path}")
