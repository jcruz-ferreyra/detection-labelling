import os
from pathlib import Path

from detection_labelling.config import DATA_DIR, HOME_DIR
from detection_labelling.utils import load_config, setup_logging

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
    "byol_dir",
    "byol_filename",
    "embed_dir",
    "embed_filename",
    "img_size",
    "batch_size",
    "sampling",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

BYOL_DIR = HOME_DIR / script_config["byol_dir"]
BYOL_FILENAME = script_config["byol_filename"]

EMBED_DIR = DATA_DIR / script_config["embed_dir"]
EMBED_FILENAME = script_config["embed_filename"]

IMG_SIZE = int(script_config["img_size"])
BATCH_SIZE = int(script_config["batch_size"])

SAMPLING = script_config["sampling"]

# Set path for input and output data
INPUT_DIR = DATA_DIR / script_config["input_folder"]
OUTPUT_DIR = DATA_DIR / script_config["output_folder"]

logger.info(f"Extracted frames main directory: {INPUT_DIR}")
logger.info(f"Selected frames main directory: {OUTPUT_DIR}")

# Select initial labelling batch
context = BatchSelectionContext(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    byol_dir=BYOL_DIR,
    byol_filename=BYOL_FILENAME,
    embed_dir=EMBED_DIR,
    embed_filename=EMBED_FILENAME,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    sampling=SAMPLING,
)

if context.train_byol:
    byol_path = BYOL_DIR / BYOL_FILENAME
    logger.info(f"No BYOL model found at {byol_path}. Initializing training.")
    train_byol(context)

if context.calculate_embed:
    embed_path = EMBED_DIR / EMBED_FILENAME
    logger.info(f"No embeddings found at {context.byol_path}. Initializing training.")
    calculate_embeddings(context)

# Task main function
select_initial_batch(context)
