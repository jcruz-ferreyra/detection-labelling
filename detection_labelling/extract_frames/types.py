from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import supervision as sv
from ultralytics import YOLO


@dataclass
class FrameExtractionContext:
    video_path: Path
    output_dir: Path
    yolo_path: Path
    yolo_params: Dict[str, Any]
    class_label: Dict[int, str]
    category_classes: Dict[str, list]
    class_confidence: List[Tuple[List[int], float]]
    frame_handler_params: Dict[str, Any]
    quadrant_scorer_params: Dict[str, Any]
    testing_frame_limit: int

    model: Optional[YOLO] = None
    generator: Optional[Generator] = None
    video_info: Optional[sv.VideoInfo] = None
    frame_handler: Optional["FrameHandler"] = None
    quadrant_scorer: Optional["QuadrantScorer"] = None

    saved_frames: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist at: {self.video_path}")


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
