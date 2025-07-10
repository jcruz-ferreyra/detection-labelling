from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the HOME and DATA paths from the .env file and convert to a Path object
HOME = Path(os.getenv("HOME")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR")).resolve()

for x in ["HOME", "DATA_DIR"]:
    path = locals()[x]
    if not path.exists():
        raise ValueError(f"{x} path '{path}' from .env does not exist.")
