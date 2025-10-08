from pathlib import Path

from detection_labelling.config import DRIVE_MODELS_DIR as MODELS_DIR
from detection_labelling.config import LOCAL_DATA_DIR as DATA_DIR
from detection_labelling.utils import check_missing_keys, load_config, setup_logging

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
    "yolo_name",
    "yolo_params",
    "class_label",
    "category_classes",
    "category_confidence",
]
check_missing_keys(required_keys, script_config)

DATASET_FOLDER = script_config["dataset_folder"]
YOLO_NAME = script_config["yolo_name"]

YOLO_PARAMS = script_config["yolo_params"]

CLASS_LABEL = {int(k): v for k, v in script_config["class_label"].items()}
CATEGORY_CLASSES = script_config["category_classes"]
CATEGORY_CONFIDENCE = script_config["category_confidence"]

CLASS_CONFIDENCE = [(CATEGORY_CLASSES[k], v) for k, v in CATEGORY_CONFIDENCE.items()]

logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Model directory: {MODELS_DIR}")

# Process video
context = ImageAnnotationContext(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    dataset_folder=DATASET_FOLDER,
    yolo_name=YOLO_NAME,
    yolo_params=YOLO_PARAMS,
    class_label=CLASS_LABEL,
    category_classes=CATEGORY_CLASSES,
    class_confidence=CLASS_CONFIDENCE,
)

# Task main function
annotate_images(context)
