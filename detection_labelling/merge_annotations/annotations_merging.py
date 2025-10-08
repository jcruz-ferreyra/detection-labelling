import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from supervision.detection.core import Detections
from supervision.utils.file import list_files_with_extensions

from detection_labelling.utils import load_pascal_voc_annotation, save_as_pascal_voc

from .types import AnnMergingContext

logger = logging.getLogger(__name__)


def _assign_confidence_scores(detections: Detections, confidence_value: float) -> Detections:
    """Assign confidence scores to detections for NMS prioritization."""
    if len(detections) > 0:
        detections.confidence = np.ones(len(detections)) * confidence_value
    return detections


def _load_and_merge_annotations_for_image(
    image_path: str, ctx: AnnMergingContext, confidence_decay: float = 0.9
) -> Detections:
    """Load annotations from all input folders for a single image and merge with NMS."""
    conf_value = 1.0
    anns = []

    for i, input_folder in enumerate(ctx.input_folders):
        logger.debug(f"Processing folder {i+1}/{len(ctx.input_folders)}: {input_folder}")

        # Decrease confidence for later folders (prioritizes earlier sources)
        conf_value *= confidence_decay

        detections, _ = load_pascal_voc_annotation(
            image_path, ctx.dataset_dir / input_folder, list(ctx.class_label.values())
        )

        # Assign confidence scores for NMS
        detections = _assign_confidence_scores(detections, conf_value)
        anns.append(detections)

        logger.debug(
            f"Loaded {len(detections)} detections from {input_folder} with confidence {conf_value:.3f}"
        )

    # Merge all annotations and apply NMS
    merged_anns = Detections.merge(anns)
    initial_count = len(merged_anns)

    if initial_count > 0:
        merged_anns = merged_anns.with_nms(class_agnostic=True)
        final_count = len(merged_anns)
        logger.debug(f"NMS reduced {initial_count} to {final_count} detections")

    return merged_anns


def _merge_annotations(ctx: AnnMergingContext) -> None:
    """Process all images and merge annotations from multiple input folders."""
    # Get all image paths
    image_paths = [
        str(path)
        for path in list_files_with_extensions(
            directory=ctx.dataset_dir / "images", extensions=["jpg", "jpeg", "png"]
        )
    ]

    logger.info(
        f"Processing {len(image_paths)} images from {len(ctx.input_folders)} annotation sources"
    )

    processed_count = 0
    total_detections = 0

    for image_path in image_paths:
        image_stem = Path(image_path).stem
        logger.debug(f"Processing image: {image_stem}")

        try:
            # Load and merge annotations for this image
            merged_anns = _load_and_merge_annotations_for_image(image_path, ctx)

            # Save merged annotations
            image = cv2.imread(image_path)
            save_as_pascal_voc(image, merged_anns, ctx.dataset_dir, image_stem, ctx.class_label, save_image=False)

            processed_count += 1
            total_detections += len(merged_anns)

            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{len(image_paths)} images")

        except Exception as e:
            logger.error(f"Failed to process image {image_stem}: {e}")
            continue

    avg_detections = total_detections / processed_count if processed_count > 0 else 0
    logger.info(
        f"Annotation merging completed: {processed_count} images processed, "
        f"average {avg_detections:.1f} detections per image"
    )


def merge_annotations(ctx: AnnMergingContext) -> None:
    """Merge annotations from multiple Pascal VOC sources with NMS for duplicate removal.

    Args:
        ctx: Context containing dataset directories, input folders, class labels, and output configuration
    """
    logger.info("Starting annotation merging process")
    logger.info(f"Input folders: {ctx.input_folders}")
    logger.info(f"Dataset directory: {ctx.dataset_dir}")
    logger.info(f"Output directory: {ctx.output_dir}")

    try:
        _merge_annotations(ctx)
        logger.info("Annotation merging completed successfully")
    except Exception as e:
        logger.error(f"Annotation merging failed: {e}")
        raise
