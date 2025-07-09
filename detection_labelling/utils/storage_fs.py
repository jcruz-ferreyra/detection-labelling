from pathlib import Path
import posixpath
import re
from typing import Callable, Tuple

import fsspec


def create_storage(config: dict) -> Tuple[fsspec.AbstractFileSystem, Callable[[str], str]]:
    """Create filesystem and path resolver from config."""
    storage_type = config["type"]
    base_data_dir = config["base_data_dir"]

    if base_data_dir is None:
        raise ValueError(f"Missing base_data_dir for storage environment: {storage_type}")

    # Create filesystem
    fs = fsspec.filesystem(storage_type, **config["fs_kwargs"])

    # Create path resolver
    if storage_type == "file":

        def resolve_path(relative_path: str) -> str:
            relative_path = _normalize_path(relative_path)
            return Path(base_data_dir) / relative_path

    else:

        def resolve_path(relative_path: str) -> str:
            relative_path = _normalize_path(relative_path)
            return f"{storage_type}://{posixpath.join(base_data_dir, relative_path)}"

    return fs, resolve_path


def _normalize_path(path: str) -> str:
    """Normalize path for cross-platform compatibility."""
    normalized = path.replace("\\", "/")
    normalized = normalized.strip("/")
    normalized = re.sub(r"/{2,}", "/", normalized)

    return normalized
