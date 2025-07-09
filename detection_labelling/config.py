from pathlib import Path
from dotenv import load_dotenv
import os

from detection_labelling.utils import create_storage

# Load environment variables from .env file
load_dotenv()

# Get the HOME and DATA paths from the .env file and convert to a Path object
HOME = Path(os.getenv("HOME")).resolve()
if not HOME.exists():
    raise ValueError(f"HOME path '{HOME}' from .env does not exist.")

# Create storage fs
ENVIRONMENT = os.getenv("ENVIRONMENT")
STORAGE_CONFIGS = {
    "development": {"type": "file", "base_data_dir": os.getenv("LOCAL_DATA_DIR"), "fs_kwargs": {}},
    "staging": {
        "type": "s3",
        "base_data_dir": os.getenv("AWS_BUCKET_NAME"),
        "fs_kwargs": {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            "use_ssl": os.getenv("MINIO_USE_SSL", "False").lower() == "true",
            "client_kwargs": {
                "verify": False,
            },
        },
    },
    "production": {
        "type": "s3",
        "base_data_dir": os.getenv("AWS_BUCKET_NAME"),
        "fs_kwargs": {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "region": os.getenv("AWS_REGION", "us-east-1"),
        },
    },
}

FS, get_full_path = create_storage(STORAGE_CONFIGS[ENVIRONMENT])
