from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from byol_pytorch import BYOL
import torch
from torch.utils.data import DataLoader


@dataclass
class BatchSelectionContext:
    input_dir: Path
    output_dir: Path
    byol_dir: Path
    byol_filename: str
    embed_dir: Path
    embed_filename: str

    img_size: int = 256
    batch_size: int = 64

    sampling: Dict = field(default_factory=dict)

    train_byol: bool = False
    calculate_embed: bool = False

    byol: Optional[BYOL] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    dataloader: Optional[DataLoader] = None
    device: Optional[torch.device] = None

    def __post_init__(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Frames directory does not exist at: {self.input_dir}")

        byol_path = self.byol_dir / self.byol_filename
        if not byol_path.is_file():
            self.train_byol = True
            byol_path.parent.mkdir(parents=True, exist_ok=True)

        embed_path = self.embed_dir / self.embed_filename
        if not embed_path.is_file():
            self.calculate_embed = True
            embed_path.parent.mkdir(parents=True, exist_ok=True)

        assert Path(self.embed_filename).suffix in [
            ".csv",
            ".parquet",
        ], "Embedding filename extension must be '.csv' or '.parquet'"

    @property
    def byol_path(self) -> Path:
        return self.byol_dir / self.byol_filename

    @property
    def byol_checkpoint_path(self) -> Path:
        filename = Path(self.byol_filename)
        return self.byol_dir / f"{filename.stem}_chp{filename.suffix}"

    @property
    def embed_path(self) -> Path:
        return self.embed_dir / self.embed_filename
