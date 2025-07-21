import os
from pathlib import Path

from detection_labelling.config import DATA_DIR, HOME_DIR
from detection_labelling.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_labelling.extract_frames import (
    FrameExtractionContext,
    extract_frames,
)

logger.info("Starting video processing pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "camera_id",
    "video_date",
    "video_num",
    "video_extension",
    "input_folder",
    "output_folder",
    "yolo_path",
    "yolo_params",
    "class_label",
    "category_classes",
    "category_confidence",
    "frame_handler_params",
    "quadrant_scorer_params",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

CAMERA_ID = script_config["camera_id"]
VIDEO_DATE = script_config["video_date"]
VIDEO_NUM = script_config["video_num"]
VIDEO_EXTENSION = script_config["video_extension"]

YOLO_PATH = HOME_DIR / script_config["yolo_path"]
YOLO_PARAMS = script_config["yolo_params"]

CLASS_LABEL = {int(k): v for k, v in script_config["class_label"].items()}
CATEGORY_CLASSES = script_config["category_classes"]
CATEGORY_CONFIDENCE = script_config["category_confidence"]

CLASS_CONFIDENCE = [(CATEGORY_CLASSES[k], v) for k, v in CATEGORY_CONFIDENCE.items()]

FRAME_HANDLER_PARAMS = script_config["frame_handler_params"]
QUADRANT_SCORER_PARAMS = script_config["quadrant_scorer_params"]

TESTING_FRAME_LIMIT = int(script_config.get("testing_frame_limit", "0"))

# Create paths for input and output directories
INPUT_DIR = DATA_DIR / script_config["input_folder"]
OUTPUT_DIR = DATA_DIR / script_config["output_folder"]

logger.info(f"Input main directory: {INPUT_DIR}")
logger.info(f"Output main directory: {OUTPUT_DIR}")

logger.info("Creating output directories")
os.makedirs(OUTPUT_DIR / "images", exist_ok=True)
os.makedirs(OUTPUT_DIR / "annotations", exist_ok=True)

# Construct video path
video_filename = f"{CAMERA_ID}_{VIDEO_DATE}_{VIDEO_NUM}{VIDEO_EXTENSION}"
video_path = INPUT_DIR / CAMERA_ID / str(VIDEO_DATE) / video_filename
logger.info(f"Video path: {video_path}")

# Process video
context = FrameExtractionContext(
    video_path=video_path,
    output_dir=OUTPUT_DIR,
    yolo_path=YOLO_PATH,
    yolo_params=YOLO_PARAMS,
    class_label=CLASS_LABEL,
    category_classes=CATEGORY_CLASSES,
    class_confidence=CLASS_CONFIDENCE,
    frame_handler_params=FRAME_HANDLER_PARAMS,
    quadrant_scorer_params=QUADRANT_SCORER_PARAMS,
    testing_frame_limit=TESTING_FRAME_LIMIT,
)

# Task main function
extract_frames(context)
