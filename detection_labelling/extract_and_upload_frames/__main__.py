import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

import yaml

from detection_labelling.config import DATA_DIR


# Set up logging
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = "extract_and_upload_frames.log"
log_path = LOG_DIR / log_filename

# Create handlers with different levels
file_handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)  # File gets DEBUG and above

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console gets INFO and above

logging.basicConfig(
    level=logging.DEBUG,  # Root logger must be at lowest level (DEBUG)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, console_handler],
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_path}")


from detection_labelling.extract_and_upload_frames import (
    VideoProcessingContext,
    process_video,
)


logger.info("Starting video processing pipeline")

# Create paths for input and output directories
INPUT_MAIN_DIR = DATA_DIR / "raw/cctv_videos"
OUTPUT_DIR = DATA_DIR / "raw/cctv_frames"

logger.info(f"Input main directory: {INPUT_MAIN_DIR}")
logger.info(f"Output main directory: {OUTPUT_DIR}")

logger.info("Creating output directories")
os.makedirs(OUTPUT_DIR / "images", exist_ok=True)
os.makedirs(OUTPUT_DIR / "annotations", exist_ok=True)

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"
logger.info(f"Loading config from: {CONFIG_PATH}")

try:
    with open(CONFIG_PATH, "r") as f:
        script_config = yaml.safe_load(f)
    logger.info(f"Successfully loaded config from {CONFIG_PATH}")
except FileNotFoundError:
    logger.error(f"Config file not found at {CONFIG_PATH}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML config: {e}")
    raise

required_keys = [
    "camera_id",
    "video_date",
    "video_num",
    "video_extension",
    "model",
    "model_params",
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

MODEL = script_config["model"]
MODEL_PARAMS = script_config["model_params"]

CLASS_LABEL = {int(k): v for k, v in script_config["class_label"].items()}
CATEGORY_CLASSES = script_config["category_classes"]
CATEGORY_CONFIDENCE = script_config["category_confidence"]

CLASS_CONFIDENCE = [(CATEGORY_CLASSES[k], v) for k, v in CATEGORY_CONFIDENCE.items()]

FRAME_HANDLER_PARAMS = script_config["frame_handler_params"]
QUADRANT_SCORER_PARAMS = script_config["quadrant_scorer_params"]

# Construct video path
video_filename = f"{CAMERA_ID}_{VIDEO_DATE}_{VIDEO_NUM}{VIDEO_EXTENSION}"
video_path = INPUT_MAIN_DIR / CAMERA_ID / str(VIDEO_DATE) / video_filename
logger.info(f"Video path: {video_path}")

# Create frame handler
context = VideoProcessingContext(
    video_path=video_path,
    output_dir=OUTPUT_DIR,
    model_name=MODEL,
    model_params=MODEL_PARAMS,
    class_label=CLASS_LABEL,
    category_classes=CATEGORY_CLASSES,
    class_confidence=CLASS_CONFIDENCE,
    frame_handler_params=FRAME_HANDLER_PARAMS,
    quadrant_scorer_params=QUADRANT_SCORER_PARAMS,
)

logger.info("Starting video processing")
process_video(context)
logger.info("Video processing completed successfully")
