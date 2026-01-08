from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

from byol_pytorch import BYOL
import torch
from torch.utils.data import DataLoader


@dataclass
class BatchSelectionContext:
    input_dir: Path
    output_dir: Path
    byol_path: Path
    embed_path: Path  # Full absolute path to embeddings file

    sampling: Dict = field(default_factory=dict)
    byol_training: Dict[str, Any] = field(default_factory=dict)

    train_byol: bool = False
    calculate_embed: bool = False

    byol: Optional[BYOL] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    dataloader: Optional[DataLoader] = None
    device: Optional[torch.device] = None

    def __post_init__(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Frames directory does not exist at: {self.input_dir}")

        if not self.byol_path.is_file():
            self.train_byol = True
            self.byol_path.parent.mkdir(parents=True, exist_ok=True)

        # Fill byol_training defaults if training is needed
        if self.train_byol:
            self.byol_training = _fill_byol_training_with_defaults(self.byol_training)

        if not self.embed_path.is_file():
            self.calculate_embed = True
            self.embed_path.parent.mkdir(parents=True, exist_ok=True)

        assert self.embed_path.suffix in [
            ".csv",
            ".parquet",
        ], "Embedding filename extension must be '.csv' or '.parquet'"

    @property
    def img_size(self) -> int:
        """Get image size for BYOL training."""
        return self.byol_training.get("img_size", 256)

    @property
    def batch_size(self) -> int:
        """Get batch size for BYOL training."""
        return self.byol_training.get("batch_size", 64)

    @property
    def byol_checkpoint_path(self) -> Path:
        return self.byol_path.parent / f"{self.byol_path.stem}_chp{self.byol_path.suffix}"


def _fill_byol_training_with_defaults(byol_training: Optional[Dict]) -> Dict:
    """Fill missing BYOL training parameters with default values."""

    defaults = {
        "img_size": 256,
        "batch_size": 64,
    }

    # If no byol_training provided at all, use all defaults
    if byol_training is None:
        return defaults

    # Merge: defaults first, then override with user values
    return {**defaults, **byol_training}
