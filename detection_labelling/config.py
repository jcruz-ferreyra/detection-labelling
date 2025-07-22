from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the HOME and DATA paths from the .env file and convert to a Path object
HOME_DIR = Path(os.getenv("HOME_DIR")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR")).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR")).resolve()

for x in ["HOME_DIR", "DATA_DIR", "MODELS_DIR"]:
    path = locals()[x]
    if not path.exists():
        raise ValueError(f"{x} path '{path}' from .env does not exist.")
