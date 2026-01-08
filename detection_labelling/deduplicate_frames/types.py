from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class FramesDeduplicationContext:
    # Required fields
    frames_dir: Path
    polygons_json_path: Path

    # Optional fields (will be filled with defaults)
    flann_index_params: Optional[Dict[str, Any]] = None
    flann_search_params: Optional[Dict[str, Any]] = None
    is_repeated_threshold: float = 0.05  # Default in field definition

    # Runtime objects
    sift: Optional[cv2.SIFT] = None
    flann: Optional[cv2.FlannBasedMatcher] = None

    def __post_init__(self):
        # Validate required paths exist
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory does not exist at: {self.frames_dir}")

        if not self.polygons_json_path.exists():
            raise FileNotFoundError(
                f"Polygons JSON file does not exist at: {self.polygons_json_path}"
            )

        # Fill defaults for optional parameters
        self.flann_index_params = _fill_flann_index_params_with_defaults(self.flann_index_params)
        self.flann_search_params = _fill_flann_search_params_with_defaults(
            self.flann_search_params
        )


def _fill_flann_index_params_with_defaults(flann_index_params: Optional[Dict]) -> Dict:
    """Fill missing FLANN index parameters with default values."""

    defaults = {
        "algorithm": 1,  # FLANN_INDEX_KDTREE
        "trees": 5,
    }

    # If no flann_index_params provided at all, use all defaults
    if flann_index_params is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **flann_index_params}


def _fill_flann_search_params_with_defaults(flann_search_params: Optional[Dict]) -> Dict:
    """Fill missing FLANN search parameters with default values."""

    defaults = {
        "checks": 50,
    }

    # If no flann_search_params provided at all, use all defaults
    if flann_search_params is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **flann_search_params}
