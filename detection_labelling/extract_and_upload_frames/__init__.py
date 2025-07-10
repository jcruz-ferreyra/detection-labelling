# extract_and_upload_frames/__init__.py
from .frame_handling import FrameHandler
from .frame_scoring import QuadrantScorer, compute_conf_weight, compute_diversity_score
from .frame_selection import get_best_frame_with_dets
from .video_processing import process_video
from .types import VideoProcessingContext

__all__ = [
    # Frame handling
    "FrameHandler",
    # Scoring functionality
    "QuadrantScorer",
    "compute_conf_weight",
    "compute_diversity_score",
    # Selection functionality
    "get_best_frame_with_dets",
    # Video processing
    "process_video",
    # Data types
    "VideoProcessingContext",
]
