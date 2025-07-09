from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from supervision import VideoInfo
from ultralytics import YOLO


@dataclass
class VideoProcessingContext:
    video_path: Path
    output_dir: str
    model: YOLO
    model_params: Dict[str, Any]
    coco_class: Dict[int, str]
    category_classes: Dict[str, list]
    category_confidence: list
    video_info: VideoInfo
    frame_generator: Any
