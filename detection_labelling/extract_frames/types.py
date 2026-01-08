from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import supervision as sv
from ultralytics import YOLO


@dataclass
class FrameExtractionContext:
    # Required fields (from config)
    video_path: Path
    output_dir: Path
    class_label: Dict[int, str]
    category_classes: Dict[str, list]

    # Optional fields (will be filled with defaults)
    yolo_path: Optional[Path] = None
    yolo_params: Optional[Dict[str, Any]] = None
    category_confidence: Optional[Dict[str, float]] = None
    frame_handler_params: Optional[Dict[str, Any]] = None
    quadrant_scorer_params: Optional[Dict[str, Any]] = None
    testing_frame_limit: Optional[int] = None

    # Derived field (created in post_init)
    class_confidence: Optional[List[Tuple[List[int], float]]] = None

    # Runtime objects
    model: Optional[YOLO] = None
    generator: Optional[Generator] = None
    video_info: Optional[sv.VideoInfo] = None
    frame_handler: Optional["FrameHandler"] = None
    quadrant_scorer: Optional["QuadrantScorer"] = None

    saved_frames: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        # Validate required paths exist
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist at: {self.video_path}")

        # Fill defaults for optional parameters
        # Note: yolo_params needs class_label, so fill it first
        self.yolo_params = _fill_yolo_params_with_defaults(self.yolo_params, self.class_label)

        # category_confidence needs category_classes and yolo_params
        self.category_confidence = _fill_category_confidence_with_defaults(
            self.category_confidence, self.category_classes, self.yolo_params
        )

        self.frame_handler_params = _fill_frame_handler_params_with_defaults(
            self.frame_handler_params
        )

        self.quadrant_scorer_params = _fill_quadrant_scorer_params_with_defaults(
            self.quadrant_scorer_params
        )

        # Fill yolo_path default
        if self.yolo_path is None:
            # Assuming MODELS_DIR is available in scope
            self.yolo_path = Path("yolov8m/yolov8m.pt")  # Will be relative to MODELS_DIR

        # Fill testing_frame_limit default
        if self.testing_frame_limit is None:
            self.testing_frame_limit = 0

        # Validate yolo model exists
        # (This check should happen in the main script where MODELS_DIR is available)

        # Create class_confidence list from category_confidence
        self.class_confidence = [
            (category_classes, self.category_confidence[category])
            for category, category_classes in self.category_classes.items()
        ]


def _fill_yolo_params_with_defaults(yolo_params: Optional[Dict], class_label: Dict) -> Dict:
    """Fill missing YOLO parameters with default values."""

    defaults = {
        "imgsz": 960,
        "verbose": False,
        "conf": 0.05,
        "iou": 0.4,
        "classes": list(class_label.keys()),  # Use all class_label keys
        "agnostic_nms": False,
    }

    # If no yolo_params provided at all, use all defaults
    if yolo_params is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **yolo_params}


def _fill_category_confidence_with_defaults(
    category_confidence: Optional[Dict], category_classes: Dict[str, list], yolo_params: Dict
) -> Dict:
    """Fill missing category confidence thresholds with default from yolo_params.conf."""

    # Use conf from yolo_params as default for all categories
    default_conf = yolo_params["conf"]

    defaults = {category: default_conf for category in category_classes.keys()}

    # If no category_confidence provided at all, use all defaults
    if category_confidence is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **category_confidence}


def _fill_frame_handler_params_with_defaults(frame_handler_params: Optional[Dict]) -> Dict:
    """Fill missing frame handler parameters with default values."""

    defaults = {
        "fps_of_interest": 0.5,
        "n_intermediate": 3,
        "seconds_skip_after_save": 2,
        "max_seconds_without_save": 10,
    }

    # If no frame_handler_params provided at all, use all defaults
    if frame_handler_params is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **frame_handler_params}


def _fill_quadrant_scorer_params_with_defaults(quadrant_scorer_params: Optional[Dict]) -> Dict:
    """Fill missing quadrant scorer parameters with default values."""

    defaults = {
        "grid_size": [4, 4],
        "k": 20,
    }

    # If no quadrant_scorer_params provided at all, use all defaults
    if quadrant_scorer_params is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **quadrant_scorer_params}


@dataclass
class ProcessingStats:
    """Statistics tracker for video frame processing."""

    frames_processed: int = 0
    frames_saved: int = 0
    frames_skipped_timing: int = 0
    frames_skipped_sampling: int = 0
    frames_no_detections: int = 0

    def update(
        self,
        *,
        processed: bool = False,
        saved: bool = False,
        skipped_timing: bool = False,
        skipped_sampling: bool = False,
        no_detections: bool = False,
    ):
        """Update statistics counters."""
        if processed:
            self.frames_processed += 1
        if saved:
            self.frames_saved += 1
        if skipped_timing:
            self.frames_skipped_timing += 1
        if skipped_sampling:
            self.frames_skipped_sampling += 1
        if no_detections:
            self.frames_no_detections += 1

    @property
    def save_rate(self) -> float:
        """Percentage of processed frames that were saved."""
        return (
            (self.frames_saved / self.frames_processed * 100) if self.frames_processed > 0 else 0.0
        )

    @property
    def total_analyzed(self) -> int:
        """Total frames that went through detection (processed + no_detections)."""
        return self.frames_processed + self.frames_no_detections

    def log_summary(self, logger, total_video_frames: int):
        """Log comprehensive statistics summary."""
        logger.info("Processing statistics:")
        logger.info(f"  - Total frames in video: {total_video_frames}")
        logger.info(f"  - Frames analyzed: {self.total_analyzed}")
        logger.info(f"  - Frames processed: {self.frames_processed}")
        logger.info(f"  - Frames saved: {self.frames_saved}")
        logger.info(f"  - Skipped (sampling): {self.frames_skipped_sampling}")
        logger.info(f"  - Skipped (timing): {self.frames_skipped_timing}")
        logger.info(f"  - No detections: {self.frames_no_detections}")
        logger.info(f"  - Save rate: {self.save_rate:.1f}%")
