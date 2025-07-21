import os
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import supervision as sv

from .types import CocoFetchingContext

logger = logging.getLogger(__name__)


def _validate_raw_output_directory(ctx: CocoFetchingContext):
    """Validate that output directory doesn't already exist."""

    if ctx.output_raw_dir.exists():
        logger.error(f"Directory already exists: {ctx.output_raw_dir}")
        raise ValueError(
            f"Directory already exists: {ctx.output_raw_dir}. Please remove it before running."
        )

    logger.info("Output directory validation passed")


def _download_coco_samples(ctx: CocoFetchingContext):
    """Download COCO samples using FiftyOne."""

    logger.info(f"Downloading COCO samples for classes: {ctx.project_classes}")

    try:
        coco_sample = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=ctx.project_classes,
            max_samples=1500,
            shuffle=True,
        )
        logger.info(f"Downloaded {len(coco_sample)} COCO samples")
        return coco_sample

    except Exception as e:
        logger.error(f"Failed to download COCO samples: {e}")
        raise


def _rename_folder_structure(ctx: CocoFetchingContext):
    """Rename folders for project compatibility."""

    logger.info("Renaming folder structure")

    rename_map = {"labels": "annotations", "data": "images"}

    try:
        for old_name, new_name in rename_map.items():
            old_path = ctx.output_raw_dir / old_name
            new_path = ctx.output_raw_dir / new_name

            if old_path.exists():
                old_path.rename(new_path)
                logger.debug(f"Renamed {old_name} to {new_name}")
            else:
                logger.warning(f"Directory not found: {old_path}")

        logger.info("Folder structure renamed successfully")

    except Exception as e:
        logger.error(f"Failed to rename folder structure: {e}")
        raise


def _export_to_voc_format(ctx: CocoFetchingContext, coco_sample):
    """Export COCO dataset to VOC format."""

    logger.info(f"Exporting to VOC format: {ctx.output_raw_dir}")

    try:
        coco_sample.export(
            export_dir=str(ctx.output_raw_dir),
            dataset_type=fo.types.dataset_types.VOCDetectionDataset,
        )
        logger.info("Export to VOC format completed")

    except Exception as e:
        logger.error(f"Failed to export to VOC format: {e}")
        raise

    _rename_folder_structure(ctx)


def _load_dataset_as_supervision(ctx: CocoFetchingContext):
    """Load the dataset using Supervision"""
    try:
        ds = sv.DetectionDataset.from_pascal_voc(
            images_directory_path=ctx.output_raw_dir / "images",
            annotations_directory_path=ctx.output_raw_dir / "annotations",
        )
        logger.info(f"Loaded dataset with {len(ds)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset as superivion: {e}")
        raise

    return ds


def _filter_underrepresented_classes(ctx: CocoFetchingContext, ds: sv.DetectionDataset):
    """Filter dataset to keep only underrepresented classes."""

    logger.info("Loading and filtering dataset for underrepresented classes")

    try:
        # Get class IDs
        project_classids = [i for i, name in enumerate(ds.classes) if name in ctx.project_classes]
        underrepr_classids = [
            i for i, name in enumerate(ds.classes) if name in ctx.underrepr_classes
        ]

        logger.info(
            f"Project classes: {len(project_classids)}, Underrepresented classes: {len(underrepr_classids)}"
        )

        # Filter dataset
        new_ds_as_list = []
        for idx, (img_path, _, annotations) in enumerate(ds):
            # Filter annotations to project classes
            annotations = annotations[np.isin(annotations.class_id, project_classids)]

            if len(annotations) == 0:
                continue

            # Keep only images with underrepresented classes
            if np.any(np.isin(annotations.class_id, underrepr_classids)):
                new_ds_as_list.append(
                    sv.DetectionDataset(
                        classes=ds.classes, images=[img_path], annotations={img_path: annotations}
                    )
                )

        logger.info(f"Filtered to {len(new_ds_as_list)} samples with underrepresented classes")

        # Merge filtered datasets
        new_ds = sv.DetectionDataset.merge(new_ds_as_list)
        logger.info(f"Merged dataset has {len(new_ds)} samples")

        return new_ds

    except Exception as e:
        logger.error(f"Failed to filter underrepresented classes: {e}")
        raise


def _save_filtered_dataset(ctx: CocoFetchingContext, filtered_ds):
    """Save the filtered dataset to interim directory."""

    logger.info(f"Saving filtered dataset to: {ctx.output_interim_dir}")

    try:
        # Create output directories
        (ctx.output_interim_dir / "images").mkdir(parents=True, exist_ok=True)
        (ctx.output_interim_dir / "annotations_oob").mkdir(parents=True, exist_ok=True)

        # Save the filtered dataset
        filtered_ds.as_pascal_voc(
            images_directory_path=ctx.output_interim_dir / "images",
            annotations_directory_path=ctx.output_interim_dir / "annotations_oob",
        )
        logger.info("Filtered dataset saved successfully")

    except Exception as e:
        logger.error(f"Failed to save filtered dataset: {e}")
        raise


def fetch_coco_samples(ctx: CocoFetchingContext):
    """
    Fetch and process COCO samples with underrepresented classes.

    Args:
        ctx: CocoFetchingContext containing configuration and paths
    """
    logger.info("Starting COCO samples fetching process")

    _validate_raw_output_directory(ctx)

    # Download COCO samples
    coco_sample = _download_coco_samples(ctx)

    # Export to VOC format
    _export_to_voc_format(ctx, coco_sample)

    # Load the dataset using Supervision
    ds = _load_dataset_as_supervision(ctx)

    # Filter for underrepresented classes
    filtered_ds = _filter_underrepresented_classes(ctx, ds)

    # Save filtered dataset
    _save_filtered_dataset(ctx, filtered_ds)

    logger.info("COCO samples fetching process completed successfully")
