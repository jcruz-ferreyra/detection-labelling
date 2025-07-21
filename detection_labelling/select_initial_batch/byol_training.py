import logging
from typing import Tuple

from byol_pytorch import BYOL
import torch
from torchvision.models import ResNet50_Weights, resnet50

from .dataset import create_image_dataloader, create_transform
from .types import BatchSelectionContext

logger = logging.getLogger(__name__)


def _initialize_model_and_optimizer(ctx: BatchSelectionContext) -> None:
    """Setup ResNet50, BYOL learner, and optimizer."""
    logger.info("Setting up BYOL learner and optimizer")

    try:
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(ctx.device)
        ctx.byol = BYOL(resnet, image_size=ctx.img_size, hidden_layer="avgpool")
        ctx.opt = torch.optim.Adam(ctx.byol.parameters(), lr=3e-4)

        logger.info("BYOL learner and optimizer initialized successfully")

    except Exception as e:
        logger.error(f"Failed to setup model and optimizer: {e}")
        raise


def _train_byol_loop(ctx: BatchSelectionContext, num_epochs: int = 100) -> None:
    """
    Execute the BYOL training loop.
    """
    logger.info(f"Starting training loop for {num_epochs} epochs")

    # Create checkpoint directory
    ctx.byol_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.byol_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Set the model to train and proceed with the model training
        ctx.byol.train()

        for epoch in range(num_epochs):
            epoch_loss = 0

            for images, _ in ctx.dataloader:
                images = images.to(ctx.device)

                loss = ctx.byol(images)
                ctx.optimizer.zero_grad()
                loss.backward()
                ctx.optimizer.step()
                ctx.byol.update_moving_average()  # Update moving average of target encoder

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(ctx.dataloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            if epoch % 10 == 0 and epoch > 0:
                torch.save(ctx.byol.net.state_dict(), str(ctx.byol_checkpoint_path))
                logger.debug(f"Saved checkpoint at epoch {epoch+1}")

        # Save final model
        torch.save(ctx.byol.net.state_dict(), str(ctx.byol_path))
        logger.info(f"Final model saved to: {ctx.byol_path}")

    except Exception as e:
        logger.error(f"BYOL training failed at epoch {epoch+1}: {e}")
        raise

    return


def train_byol(ctx: BatchSelectionContext) -> None:
    """
    Train BYOL model on images from the frames directory.

    Args:
        ctx: BatchSelectionContext containing training configuration and paths
    """
    logger.info("Starting BYOL training process")

    # Set up device
    ctx.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {ctx.device}")

    # Create data transforms and dataloader
    transform = create_transform()
    ctx.dataloader = create_image_dataloader(ctx.input_dir / "images", ctx.batch_size, transform)

    # Initialize BYOL model and optimizer and add them to context.
    _initialize_model_and_optimizer(ctx)

    # Train the model
    _train_byol_loop(ctx, num_epochs=100)

    logger.info("BYOL training process completed successfully")
