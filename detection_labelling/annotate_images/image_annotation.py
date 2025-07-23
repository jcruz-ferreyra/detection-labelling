import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from detection_labelling.utils import get_ultralytics_detections, save_as_pascal_voc

from .types import ImageAnnotationContext

logger = logging.getLogger(__name__)


def _initialize_model(ctx: ImageAnnotationContext) -> None:
    """Initialize YOLO model. Add it to context."""
    logger.info(f"Loading YOLO model: {ctx.yolo_path}")

    try:
        ctx.model = YOLO(ctx.yolo_path)
        ctx.model.fuse()
        logger.info(f"Successfully loaded and fused YOLO model: {ctx.yolo_path}")

    except Exception as e:
        logger.error(f"Failed to load YOLO model {ctx.yolo_path}: {e}")
        raise


def _get_image_detections(
    ctx: ImageAnnotationContext, frame: np.ndarray, filename: str
) -> Optional[sv.Detections]:
    """Get YOLO detections for single image."""
    logger.debug(f"Running detection on image {filename}")

    try:
        dets = get_ultralytics_detections(
            frame, ctx.model, ctx.yolo_params, ctx.class_confidence, bgr=True
        )
        logger.debug(f"Found {len(dets)} detections in frame {filename}")
        return dets

    except Exception as e:
        logger.warning(f"Detection failed on frame {filename}: {e}")
        return sv.Detections.empty()


def _get_and_save_annotations(ctx: ImageAnnotationContext):
    """Process all images in the dataset directory and save annotations."""
    images_dir = ctx.dataset_dir / "images"

    # Check if images directory exists
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found at: {images_dir}\n"
            f"Please ensure your images are located in: {ctx.dataset_dir}/images/"
        )

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in: {images_dir}\n"
            f"Please place your images (.png, .jpg, .jpeg, .bmp, .tiff) in: {ctx.dataset_dir}/images/"
        )

    logger.info(f"Found {len(image_files)} images to process in {images_dir}")

    annotations_dir = ctx.dataset_dir / "annotations_oob"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Annotations will be saved to: {annotations_dir}")

    for idx, image_filename in enumerate(image_files):
        image_path = images_dir / image_filename
        logger.debug(f"Processing image {idx + 1}/{len(image_files)}: {image_filename}")

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            continue

        # Get detections for the current frame
        dets = _get_image_detections(ctx, image, image_filename)
        if dets is None:
            continue

        filename_stem = Path(image_filename).stem

        try:
            save_as_pascal_voc(
                image, dets, ctx.dataset_dir, filename_stem, ctx.class_label, save_image=False
            )
            logger.debug(f"Saved annotations for {filename_stem}")
        except Exception as e:
            logger.error(f"Failed to save annotations for {filename_stem}: {e}")
            continue

    logger.info(f"Completed processing {len(image_files)} images")


def annotate_images(ctx: ImageAnnotationContext):
    """
    Process images to detect objects and save annotations in Pascal VOC format.

    Args:
        ctx: ImageAnnotationContext containing all necessary parameters and objects
    """
    logger.info(f"Starting image annotation process for dataset: {ctx.dataset_dir}")

    # Initialize model
    _initialize_model(ctx)

    # Perform inference and save preliminary annotations
    _get_and_save_annotations(ctx)

    logger.info("Image annotation process completed successfully")
