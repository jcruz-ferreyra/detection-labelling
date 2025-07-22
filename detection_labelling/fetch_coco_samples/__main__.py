import os
from pathlib import Path

from detection_labelling.config import DATA_DIR, HOME_DIR
from detection_labelling.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_labelling.fetch_coco_samples import (
    CocoFetchingContext,
    fetch_coco_samples,
)

logger.info("Starting coco samples fetching pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = ["output_raw_folder", "output_interim_folder", "project_classes", "underrepresented_classes"]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

PROJECT_CLASSES = script_config["project_classes"]
UNDERREPR_CLASSES = script_config["underrepresented_classes"]

# Create paths for frames directory
OUTPUT_RAW_DIR = DATA_DIR / script_config["output_raw_folder"]
OUTPUT_INTERIM_DIR = DATA_DIR / script_config["output_interim_folder"]

logger.info(f"Extracted frames main directory: {OUTPUT_RAW_DIR}")
logger.info(f"Selected frames main directory: {OUTPUT_INTERIM_DIR}")


# Select initial labelling batch
context = CocoFetchingContext(
    output_raw_dir=OUTPUT_RAW_DIR,
    output_interim_dir=OUTPUT_INTERIM_DIR,
    project_classes=PROJECT_CLASSES,
    underrepr_classes=UNDERREPR_CLASSES,
)

# Task main function
fetch_coco_samples(context)
