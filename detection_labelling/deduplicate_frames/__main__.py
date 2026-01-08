from pathlib import Path

from detection_labelling.config import LOCAL_DATA_DIR as DATA_DIR
from detection_labelling.utils import check_missing_keys, load_config, setup_logging

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
]
check_missing_keys(required_keys, script_config)

# Required parameters
FRAMES_FOLDER = script_config["frames_folder"]
POLYGONS_JSON_FILENAME = script_config["polygons_json_filename"]

POLYGONS_JSON_PATH = Path(__file__).parent.resolve() / POLYGONS_JSON_FILENAME

# Optional parameters - use .get()
FLANN_INDEX_PARAMS = script_config.get("flann_index_params")
FLANN_SEARCH_PARAMS = script_config.get("flann_search_params")
IS_REPEATED_THRESHOLD = script_config.get("is_repeated_threshold", 0.05)

# Create paths for frames directory
FRAMES_DIR = DATA_DIR / FRAMES_FOLDER

logger.info(f"Extracted frames directory: {FRAMES_DIR}")
logger.info(f"Polygons JSON path: {POLYGONS_JSON_PATH}")

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
