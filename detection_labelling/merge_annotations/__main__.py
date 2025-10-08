from pathlib import Path

from detection_labelling.config import DRIVE_MODELS_DIR as MODELS_DIR
from detection_labelling.config import LOCAL_DATA_DIR as DATA_DIR
from detection_labelling.utils import check_missing_keys, load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_labelling.merge_annotations import AnnMergingContext, merge_annotations

logger.info("Starting video processing pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "dataset_folder",
    "input_folders",
    "class_label",
]
check_missing_keys(required_keys, script_config)

DATASET_FOLDER = script_config["dataset_folder"]
INPUT_FOLDER = script_config["input_folders"]

CLASS_LABEL = {int(k): v for k, v in script_config["class_label"].items()}

logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Model directory: {MODELS_DIR}")

# Process video
context = AnnMergingContext(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    dataset_folder=DATASET_FOLDER,
    input_folders=INPUT_FOLDER,
    class_label=CLASS_LABEL,
)

# Task main function
merge_annotations(context)
