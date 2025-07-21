# extract_and_upload_frames/__init__.py
from .frame_extraction import extract_frames
from .frame_handling import FrameHandler
from .frame_scoring import QuadrantScorer, compute_conf_weight, compute_diversity_score
from .types import FrameExtractionContext, ProcessingStats

__all__ = [
    # Frame handling
    "FrameHandler",
    # Scoring functionality
    "QuadrantScorer",
    "compute_conf_weight",
    "compute_diversity_score",
    # Video processing
    "extract_frames",
    # Data types
    "FrameExtractionContext",
    "ProcessingStats",
]
