import logging
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


def create_transform(img_size: int = 256) -> transforms.Compose:
    """Create image transformations for BYOL training."""
    logger.info(f"Setting up image transformations ({img_size}x{img_size} resize)")
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_image_dataloader(root_dir: Path, batch_size: int, transform: transforms.Compose) -> DataLoader:
    """Create dataloader for processing images in batches."""
    logger.info(f"Creating dataloader from: {root_dir}")

    try:
        dataset = ImageDataset(root_dir=root_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Dataset loaded: {len(dataset)} images, {len(dataloader)} batches")
        return dataloader

    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        raise
