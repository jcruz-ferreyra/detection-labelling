from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CocoFetchingContext:
    output_raw_dir: Path
    output_interim_dir: Path
    project_classes: List
    underrepr_classes: List
