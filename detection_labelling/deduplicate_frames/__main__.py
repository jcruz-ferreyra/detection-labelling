from pathlib import Path

from detection_labelling.config import DATA_DIR
from detection_labelling.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_labelling.deduplicate_frames import (
    FramesDeduplicationContext,
    deduplicate_frames,
)

logger.info("Starting frame deduplication pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "frames_folder",
    "polygons_json_filename",
    "flann_index_params",
    "flann_search_params",
    "is_repeated_threshold",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

FLANN_INDEX_PARAMS = script_config["flann_index_params"]
FLANN_SEARCH_PARAMS = script_config["flann_search_params"]
POLYGONS_JSON_PATH = Path(__file__).parent.resolve() / script_config["polygons_json_filename"]
IS_REPEATED_THRESHOLD = script_config["is_repeated_threshold"]

# Create paths for frames directory
FRAMES_DIR = DATA_DIR / script_config["frames_folder"]
logger.info(f"Extracted frames directory: {FRAMES_DIR}")

# Deduplicate frames
context = FramesDeduplicationContext(
    frames_dir=FRAMES_DIR,
    flann_index_params=FLANN_INDEX_PARAMS,
    flann_search_params=FLANN_SEARCH_PARAMS,
    polygons_json_path=POLYGONS_JSON_PATH,
    is_repeated_threshold=IS_REPEATED_THRESHOLD,
)

# Task main function
deduplicate_frames(context)
