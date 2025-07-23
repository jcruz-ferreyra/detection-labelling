from pathlib import Path

from detection_labelling.config import DATA_DIR, MODELS_DIR
from detection_labelling.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_labelling.annotate_images import ImageAnnotationContext, annotate_images

logger.info("Starting video processing pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "dataset_folder",
    "yolo_path",
    "yolo_params",
    "class_label",
    "category_classes",
    "category_confidence",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

YOLO_PATH = MODELS_DIR / script_config["yolo_path"]
YOLO_PARAMS = script_config["yolo_params"]

CLASS_LABEL = {int(k): v for k, v in script_config["class_label"].items()}
CATEGORY_CLASSES = script_config["category_classes"]
CATEGORY_CONFIDENCE = script_config["category_confidence"]

CLASS_CONFIDENCE = [(CATEGORY_CLASSES[k], v) for k, v in CATEGORY_CONFIDENCE.items()]

# Create paths for input and output directories
DATASET_DIR = DATA_DIR / script_config["dataset_folder"]
logger.info(f"Images main directory: {DATASET_DIR}")

# Process video
context = ImageAnnotationContext(
    dataset_dir=DATASET_DIR,
    yolo_path=YOLO_PATH,
    yolo_params=YOLO_PARAMS,
    class_label=CLASS_LABEL,
    category_classes=CATEGORY_CLASSES,
    class_confidence=CLASS_CONFIDENCE,
)

# Task main function
annotate_images(context)
