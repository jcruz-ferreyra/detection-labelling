import os
from pathlib import Path

from detection_labelling.config import LOCAL_DATA_DIR as DATA_DIR
from detection_labelling.config import LOCAL_MODELS_DIR as MODELS_DIR
from detection_labelling.utils import check_missing_keys, load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_labelling.select_initial_batch import (
    BatchSelectionContext,
    calculate_embeddings,
    select_initial_batch,
    train_byol,
)

logger.info("Starting initial batch selection pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "input_folder",
    "output_folder",
    "byol_filename",
    "embed_filename",
    "sampling",
]
check_missing_keys(required_keys, script_config)

BYOL_PATH = MODELS_DIR / script_config["byol_filename"]
EMBED_PATH = DATA_DIR / script_config["embed_filename"]
SAMPLING = script_config["sampling"]

BYOL_TRAINING = script_config.get("byol_training", {})

# Set path for input and output data
INPUT_DIR = DATA_DIR / script_config["input_folder"]
OUTPUT_DIR = DATA_DIR / script_config["output_folder"]

logger.info(f"Extracted frames main directory: {INPUT_DIR}")
logger.info(f"Selected frames main directory: {OUTPUT_DIR}")

# Select initial labelling batch
context = BatchSelectionContext(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    byol_path=BYOL_PATH,
    embed_path=EMBED_PATH,
    byol_training=BYOL_TRAINING,
    sampling=SAMPLING,
)

if context.train_byol:
    logger.info(f"No BYOL model found at {context.byol_path}. Initializing training.")
    train_byol(context)

if context.calculate_embed:
    logger.info(f"No embeddings found at {context.embed_path}. Initializing calculation.")
    calculate_embeddings(context)

# Task main function
select_initial_batch(context)
