import logging
from pathlib import Path

from byol_pytorch import BYOL
import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from .dataset import create_image_dataloader, create_transform
from .types import BatchSelectionContext

logger = logging.getLogger(__name__)


def _initialize_model(ctx: BatchSelectionContext) -> BYOL:
    """Setup ResNet50 and BYOL learner with pretrained weights."""
    logger.info(f"Setting up BYOL learner with pretrained weights from: {ctx.byol_path}")

    try:
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(ctx.device)
        resnet.load_state_dict(torch.load(ctx.byol_path, map_location=ctx.device))
        logger.info("Pretrained weights loaded successfully")

        ctx.byol = BYOL(resnet, image_size=ctx.img_size, hidden_layer="avgpool")
        logger.info("BYOL learner initialized")

    except FileNotFoundError:
        logger.error(f"Pretrained model not found at: {ctx.byol_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to setup BYOL model: {e}")
        raise


def _calculate_embeddings_loop(ctx: BatchSelectionContext):
    """Process all images through model and save embeddings to file."""
    logger.info("Starting embedding calculation loop")

    paths_list = []
    embeddings_list = []

    try:
        # Set model to eval and calculate the embeddings
        ctx.byol.eval()

        with torch.no_grad():
            for images, paths in tqdm(ctx.dataloader, desc="Calculating Embeddings"):
                images = images.to(ctx.device)

                _, embedding = ctx.byol(images, return_embedding=True)
                embedding = embedding.cpu().numpy()

                # Store batch results
                paths_list.extend(paths)
                embeddings_list.append(embedding)

        # Stack all embeddings
        embeddings_array = np.vstack(embeddings_list)  # shape: [N_images, embedding_dim]
        logger.info(f"Calculated embeddings shape: {embeddings_array.shape}")

        # Extract filenames
        filenames_list = [Path(path).name for path in paths_list]

        # Save based on file extension
        ctx.embed_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(embeddings_array)
        df["filename"] = filenames_list

        suffix = ctx.embed_path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(ctx.embed_path, index=False)
            logger.info(f"Embeddings saved to CSV: {ctx.embed_path}")
        elif suffix == ".parquet":
            df.to_parquet(ctx.embed_path, index=False)
            logger.info(f"Embeddings saved to Parquet: {ctx.embed_path}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    except Exception as e:
        logger.error(f"Failed during embedding calculation: {e}")
        raise


def calculate_embeddings(ctx: BatchSelectionContext) -> None:
    """
    Calculate embeddings for images using pretrained BYOL model.

    Args:
        ctx: BatchSelectionContext containing paths, model config, and runtime objects
    """
    logger.info("Starting embedding calculation process")

    # Set up device
    ctx.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {ctx.device}")

    # Create data transforms and dataloader
    transform = create_transform()
    ctx.dataloader = create_image_dataloader(ctx.input_dir / "images", ctx.batch_size, transform)

    # Load pretrained BYOL model and add it to context
    _initialize_model(ctx)

    # Calculate embeddings for all images
    _calculate_embeddings_loop(ctx)

    logger.info("Embedding calculation process completed successfully")
